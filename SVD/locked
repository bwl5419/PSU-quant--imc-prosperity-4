## SVD rank selection Simulation
## Math 414h Final Paper

## Our goal is to use probability(random matirces + null distributions) to decide
## how many dimensions(r) to keep when doing SVD on a data matrix

## part 1: Build the null distrb to figure out what the singular values look like for pure boise
## part 2: False postiive calibraiton so if there is actually no singal, does my method still wrongly say there is a signal
## part 3: Can we actually detect real strcuture when it exists


import numpy as np  ## handles matrices and that
import matplotlib.pyplot as plt ## makes plots and such

def largest_singular_value(matrix):
    
    
    """
    Takes a 2D numpy matrix as our input 

    Returns the single largest singular value as a float bc that's the only one we care about

    np.linalg.svd() does the full SVD decomposition.
    compute_uv=False means we only want the singular values and thus not U or V
    Singular values come back sorted largest to smallest, so our index [0] would be the biggest

    
    """
    singular_values= np.linalg.svd(matrix,compute_uv=False)
    return singular_values[0]


## now use out helper to estimate rank using the edge rule( which is just out way of choosing how many singular vlaues are "real")

def estimate_rank(matrix,threshold):
    """
    
    Takes a matrix and a threshold (which for us will be the 95th percentile cutoff).
    Returns r_hat: the number of singular values above the threshold.
 
    This is the core rank estimator from the paper:
        r_hat = # { k : s_k(X) > threshold }
        s_k(x) is the just the kth singular value
        so it's just the kth singular value is bigger than the cutoff that we chose
    """

    
    singular_values = np.linalg.svd(matrix,compute_uv=False)

    count = 0 
    for sv in singular_values:
        if sv> threshold:
            count= count +1
    return count 

## setting our matrix size and simulaiton count

m=20 ## our row count which is the number of dimensions
n=50 ## our colum count or our number of products
n_sim= 1000 ## this is the num of random matrices we're simulating

## part 1: Making the null distrb which is if x were pure noise, how large would that svd b
## so we're gonna simulate 1000 random matrices and record the largest sv each time

print("=" * 50) ## for organiaztion and making it clean looking
print("part 1:building the null distr")
print("=" * 50)

null_max_sv_list= [] ## this will have the largest sv from each sim

for i in range(n_sim):
    ## we gotta fen a random noise matrix
    ## thus each entry is drawn indepdently from N(0,1)
    # this is our prob model for noise

    E = np.random.rand(m,n)

    ## now we get the largest singular value

    max_sv= largest_singular_value(E)

    null_max_sv_list.append(max_sv)

    ## we gotta conver the list to a numpy array so we can easily do the math on it

null_max_sv= np.array(null_max_sv_list)

    ## step 4 is to calc the 95th percentile so that 95% of ure noise matrices fall below this value
    ## thus if smth exceed this, it's prob real
q95= np.percentile(null_max_sv_list,95)

print(f"Mean of null max singular values:   {np.mean(null_max_sv):.4f}")
print(f"Std dev of null max singular values: {np.std(null_max_sv):.4f}")
print(f"95th percentile (our threshold q95): {q95:.4f}") ## 4f is a float with 4 digits
print()
## figure 1 -> histogram of null distr

plt.figure(figsize=(7,4))
plt.hist(null_max_sv_list, bins=40, color ="steelblue", edgecolor= "white", alpha=0.85)
plt.axvline(q95, color = "red", linewidth=2, linestyle = "--", label=f"q95 = {q95:.2f}")
plt.xlabel("largest singular value")
plt.ylabel("frequence(from 1000 sims)")
plt.title("Null Distribution of the Largest Singular Value\n(pure Gaussian noise, m=20, n=50)")
plt.legend()
plt.tight_layoutplt.savefig("figure1_null_distribution.png", dpi=150)
plt.show( )
print("Figure 1 saved: figure1_null_distribution.png")
print()

## false positive calibration
## does our threshold actuallly control false postives
 
