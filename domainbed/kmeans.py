import torch
import numpy as np

def pairwise_distance(data1, data2 = None, device=-1):
    if data2 is None:
        data2 = data1
    if device != -1:
        data1, data2 = data1.cuda(device), data2.cuda(device)

    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)

    dis = (A-B)**2.0

    dis = dis.sum(dim=-1).squeeze()
    return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
        for group_index_c, group_c in enumerate(groups):
            R, C = X[group_r], X[group_c]
            if device != -1:
                R = R.cuda(device)
                C = C.cuda(device)

            group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict

def forgy(X, n_clusters):
    _len = len(X)
    while True:
        indices = np.random.choice(_len, n_clusters)
        initial_state = X[indices]
        # if np.abs(indices[0] - indices[1])>0.5:
            # break
        indices_uniq = np.unique(indices)
        if len(indices_uniq) == len(indices):
            break
    return initial_state

def lloyd(X, n_clusters, device=0, tol=1e-4, max_step=10):
    assert not torch.isnan(X).any(), 'X is nan'
    
    initial_state = forgy(X, n_clusters)
    assert not torch.isnan(initial_state).any(), 'initial_state is nan'
    error_c = 0
    step = 1
    
    while True:

        dis = pairwise_distance(X, initial_state)
        choice_cluster  = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre)**2, dim=1)))
        if center_shift**2 < tol:
            break
        step = step + 1
        if step > max_step:
            break
        if torch.isnan(initial_state).any():
            initial_state = forgy(X, n_clusters)
            step = 1
            error_c += 1
            if error_c >= 20: # after more than 20 unsuccessful clustering attempts, reduce the cluster numbers
                return None, None

    return choice_cluster, initial_state


if __name__=='__main__':

    import numpy as np 
    A = np.concatenate([np.random.randn(1000, 2), np.random.randn(1000, 2)+3, np.random.randn(1000, 2)+6], axis=0)
    clusters_index, centers = lloyd(A, 2, device=0, tol=1e-4)

    print(clusters_index)
