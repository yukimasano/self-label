import time
import torch
from util import MovingAverage

def aggreg_multi_gpu(model, dataloader, hc, dim, TYPE=torch.float64, model_gpus=1):
    """"Accumulate activations and save them on multiple GPUs
        * this function assumes the model is on the first `model_gpus` GPUs
          so that it can write the activations on the remaining ones
        * it splits the activations evenly between the remaining GPUs
    """
    # number of gpus to store
    ngpu_store = torch.cuda.device_count() - model_gpus

    # number of batches in DL
    l_dl = len(dataloader)

    # number of batches each gpu gets
    batches_per_gpu = l_dl // ngpu_store

    # number of data each gpu gets
    points_per_gpu = batches_per_gpu*dataloader.batch_size

    # empty array of indices that we need to keep track of
    indices = torch.empty(len(dataloader.dataset), dtype=torch.long)

    # set up matrix PS: (N x K) when using one head, otherwise N x D, where D is the dim before the last FC layer.
    PS = [torch.empty(points_per_gpu, dim,
                      device='cuda:' + str(i), dtype=TYPE)
          for i in range(model_gpus, model_gpus + ngpu_store-1)]
    # accomodate remainder
    PS.append(torch.empty(len(dataloader.dataset) - (ngpu_store-1)*points_per_gpu,
                          dim, device='cuda:' + str(model_gpus + ngpu_store - 1), dtype=TYPE))

    # slice sizes, i.e. how many activations will be on the gpus
    slices = [qq.shape[0] for qq in PS]
    print("slice sizes: ", slices, flush=True)
    batch_time = MovingAverage(intertia=0.9)
    now = time.time()
    st = 0
    softmax = torch.nn.Softmax(dim=1).to('cuda:0')

    # switch the model to not output array but instead last-FC output for one head and pre-last activations for multi-heads
    model.headcount = 1
    for batch_idx, (data, _, _selected) in enumerate(dataloader):
        data = data.to(torch.device('cuda:0'))
        mass = data.size(0)
        en = st + mass
        # j keeps track of which part of PS we're writing to
        j = min((batch_idx // batches_per_gpu), ngpu_store - 1)
        subs = j*points_per_gpu
        if hc == 1:
            p = softmax(model(data)).detach().to(TYPE)
            # when using one head: save softmax (N x K) matrix:
            PS[j][st-subs:en-subs, :].copy_(p)
        else:
            # when using multiple heads: save softmax (N x D) matrix
            PS[j][st-subs:en-subs, :].copy_(model(data).detach())
        indices[st:en].copy_(_selected)
        st = en
        batch_time.update(time.time() - now)
        now = time.time()
        if batch_idx % 50 == 0:
            print(f"Aggregating batch {batch_idx:03}/{l_dl}, speed: {mass / batch_time.avg:04.1f}Hz. To rGPU {j+1}",
                  end='\r', flush=True)
    torch.cuda.synchronize() # just in case
    return PS, indices


def gpu_mul_Ax(A, b, ngpu, splits, TYPE=torch.float64,model_gpus=1):
    """ multiplies matrix A (stored on multiple GPUs) with vector x
        * returns vector on GPU 0
    """
    # Step 1: make a copy of B on each GPU
    N = splits[-1]
    b_ = []
    for i in range(model_gpus,  ngpu):
        b_.append(b.to('cuda:' + str(i)))
    # Step 2: issue the matmul on each GPU
    c = torch.empty(N, 1, device='cuda:0', dtype=TYPE)
    for a,i in enumerate(range(model_gpus,  ngpu)):
        c[splits[a]:splits[a+1], :].copy_(torch.matmul(A[a], b_[a]))
    return c


def gpu_mul_AB(A, B, c, dim, TYPE=torch.float64, model_gpus=1):
    """" multiplies to matrices A,B on GPU and adds vector c and does softmax at the end
         * used to compute the effect of a linear FC layer followed by softmax
         * return (N x K) matrix spread over the same GPUs as the PS matrix
    """
    # Step 1: make a copy of B on each GPU
    ngpu = torch.cuda.device_count()  # one for the model
    b_ = []
    for i in range(model_gpus, ngpu):
        b_.append(B.to('cuda:' + str(i)))
    # Step 2: issue the matmul on each GPU
    PS = []
    for a, i in enumerate(range(model_gpus, ngpu)):
        PS.append((torch.matmul(A[a], b_[a]) + c.to('cuda:'+str(i))).to(TYPE))
        # the softmax
        torch.exp(PS[a], out=PS[a])
        summed = torch.sum(PS[a], dim=1, keepdim=True)
        PS[a] /= summed
    return PS


def gpu_mul_xA(b, A, ngpu, splits, TYPE=torch.float64, model_gpus=1):
    """ multiplies vector x with matrix A (stored on multiple GPUs)
            * returns vector on GPU 0
    """
    # Step 1: make a copy of B on each GPU
    b_ = []
    for a, i in enumerate(range(model_gpus, ngpu)):
        b_.append(b[:, splits[a]:splits[a+1]].to('cuda:' + str(i)))
    # Step 2: issue the matmul on each GPU
    c = torch.empty(ngpu-model_gpus, A[0].size(1), device='cuda:0', dtype=TYPE)
    for a, i in enumerate(range(model_gpus,  ngpu)):
        c[a:a+1, :].copy_(torch.matmul(b_[a], A[a]))
    # Step 3: need to sum these up
    torch.cuda.synchronize() # just in case
    c = torch.sum(c, 0, keepdim=True)
    return c
