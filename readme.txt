Readme for Multi Object Compressive Tracking

By:
Mohit Minhas
P. Deepak Prashanth

Mentor:
Dr. Jagdish Raheja

---------------------------------------------------------------------------------------------------------------------------------------------
Procedure
> specify the video path in RuntrackerMulti
> run mexCompile.m to generate mex files 
> run RuntrackerMulti.m
> specify number of objects through command line
> select bounding boxes
The parameters in the main function ¡°Runtracker.m¡± can be tuned as follows
1.	¡°trparams.init_postrainrad¡± is the search radius for the positive sample; This parameter can be set 4~8. If the object moves very fast, a large parameter should be used to contain more positive samples.
2.	¡°trparams.srchwinsz¡± is the search radius for the search window at the new frame; This parameter can be set 15~35. If the object moves fast, a larger parameter should be used to contain the object.
3.	¡°lRate¡± is the learning rate parameter. This parameter can be set 0.7~0.95; If the object changes fast, a small ¡°lRate¡± should be used to weight more on the new frames.
4.	¡°ftrparams.maxNumRect¡± is the maximum number of nonzero entries at each row of the random matrix. The parameter can be set 4 or 6. If the appearance of the object varies much, 6 should be used to contain more discriminative features.


Reference
---
RealTime
Compressive Tracking, Kaihua Zhang(a), Lei Zhang(a), and MingHsuan
Yang(b),
(a) Depart. of Computing, Hong Kong Polytechnic University, (b) Electrical Engineering and
Computer Science, University of California at Merced
---