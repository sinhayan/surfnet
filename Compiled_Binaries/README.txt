You can find detailed and up to date documentation here:
	http://www.cs.princeton.edu/~vk/CorrsCode/doc_bin.html

Make sure you have downloaded the most recent code version.

Code by: Vladimir G. Kim, Thomas Funkhouser, Yaron Lipman
	Princeton University
Correspond via: vk@princeton.edu 

This folder contains executables to find inter-surface maps and
correspondences as described in:
	[1] Blended Intrinsic Maps.
		Vladimir G. Kim, Yaron Lipman, Thomas Funkhouser.
		Transaction on Graphics (Proc. SIGGRAPH 2011)
		
	[2] Mobius Transformations For Global Intrinsic Symmetry Analysis
		Vladimir Kim and Yaron Lipman and Xiaobai Chen and Thomas Funkhouser
		Computer Graphics Forum (Proc. of SGP 2010) 

	[3] Mobius Voting for Surface Correspondence. 
		Yaron Lipman and Thomas Funkhouser. 
		ACM Transactions on Graphics (Proc. SIGGRAPH 2009)

DISCLAIMER
These executables are provided as is without any warranty. 
They WERE used to produce results for 'Blended Intrinsic Maps' project,
but WERE NOT used for the 'Mobius Voting' project and thus might not
reflect all the details of the original implementation (if you want 
the original implementation of Mobius Voting you should contact Yaron Lipman).

Disadvantages of this implementation include (but not limited to): 
    * For intrinsic symmetry detection only triplets of correspondences are searched (original version included search for quadruplets, and fit average mobius to them), so the feature points have to include the stationary point. 

USAGE
Just run a script to see a list of parameters.
You can modify three parameters in the script: 
	settings*.txt - contains algorithm parameters.
	render*.txt - contains rendering parameters.


