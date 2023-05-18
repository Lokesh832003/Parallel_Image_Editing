# Parallel Image Editing Algorithms
In this project, the different edited forms (mirrored, negated, grayscaled etc.) of a selected image are produced. This project makes use of OpenMP to parallelize the image processing required to edit images using multiple threads. Each thread will do the same function but on different parts of the image to decrease the time needed for getting the final product.   
  
The different integrated functions are -
<ul>
  <li>Gaussian blur</li>
  <li>Mirror function</li>
  <li>The Negation function</li>
  <li>Grayscaling</li>
  <li>Saturation</li>
</ul>

To compare the time difference between the serial implementation and the parallel implementation, the total time taken for both of the implementations are compared.  
   
<i>Serial Implementation:</i>  
![image](https://github.com/Lokesh832003/Parallel_Image_Editing/assets/121274778/1fb5abe8-4fcf-4d39-88ce-9401779cd4e3)

<i>Parallel Implementation:</i>  
![image](https://github.com/Lokesh832003/Parallel_Image_Editing/assets/121274778/4a080a75-2edb-4788-b7f4-3b9da77a20fd)

As you can see, the time taken for the parallel implementation is 50% faster than the serail implementation. And this comparison was done for a 256x256 image. This time difference will only increase with the image size, making the parallel implementaion much more desirable.
