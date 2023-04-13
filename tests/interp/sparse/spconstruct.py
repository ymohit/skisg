import torch

from skisg.config import SgBasisType
from skisg.interp.sparse.sgindices import compute_LI_pairs, compute_LI


def helptest_LI_pairs(grid_level, ndim, basis=SgBasisType.NAIVE):

    LIs = compute_LI_pairs(grid_level, ndim, basis=basis)
    
    for d in range(1, ndim+1):
        LI_mat = compute_LI(grid_level, d, basis=basis)
        LI_mat = torch.from_numpy(LI_mat).to(torch.int32)

        for gl in range(grid_level+1):
            desired_mat = torch.sum(LI_mat[:, ::2], axis=1) <= gl
            assert torch.equal(desired_mat, LIs[(gl, d)])
            
            
def test_LI_pairs(basis=SgBasisType.NAIVE):
    helptest_LI_pairs(grid_level=0, ndim=2, basis=basis)
    helptest_LI_pairs(grid_level=5, ndim=4, basis=basis)
    helptest_LI_pairs(grid_level=3, ndim=4, basis=basis)
    helptest_LI_pairs(grid_level=2, ndim=4, basis=basis)
    helptest_LI_pairs(grid_level=2, ndim=3, basis=basis)
    helptest_LI_pairs(grid_level=1, ndim=3, basis=basis)
    helptest_LI_pairs(grid_level=0, ndim=3, basis=basis)
    helptest_LI_pairs(grid_level=0, ndim=4, basis=basis)
    helptest_LI_pairs(grid_level=2, ndim=2, basis=basis)
    helptest_LI_pairs(grid_level=1, ndim=2, basis=basis)


if __name__ == '__main__':
    test_LI_pairs(basis=SgBasisType.NAIVE)
    test_LI_pairs(basis=SgBasisType.BOUNDSTART)
