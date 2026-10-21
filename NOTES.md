https://en.wikipedia.org/wiki/X-ray_transform 
https://en.wikipedia.org/wiki/Radon_transform

plucker vs ray space model
Ray-space: $\Phi(\ell,t)$ with $\ell$ as input
Plucker space: $\Phi(\mathbf d, \mathbf m, t)$ with $(\mathbf d, \mathbf m)$ as input


use accumulation in x and y to move around control points for CDF 

default strategy:
(o, d) -> same as normal NeRF

* output CDF would be [0,1] where 1 is the full size of the world

* output grounded in real world but input is ungrounded

* need to have direction as input for training since we need to be able to query this at arbitrary directions at render time

*

Alternate strategy:
(o, p) -> pass in origin and arbitary point of collision

* output is normalized to [0, p]

* grounds us in 3d geometry for the input, output is not very grounded

* augment training data to get (o, 1.1*p), (o, 1.2*p), etc. with identical CDF output to (o,p), except more time spent at 1 at the end 

* also add training data to get (o, 0.9*p) , etc with all zeros in the CDF
