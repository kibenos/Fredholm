An example of using CUDA technology for the numerical solution of the [Fredholm integral equation of the second kind](https://en.wikipedia.org/wiki/Fredholm_integral_equation#Equation_of_the_second_kind):

<p align="center">
  <img src="img/Fredholmeq.png" alt="drawing" width="450"/>
</p>

# Algorithm

The integral is approximated by a compound [quadrature formula](https://en.wikipedia.org/wiki/Numerical_integration#Quadrature_rules_based_on_interpolating_functions) of order m:

<p align="center">
  <img src="img/quadrature.png" alt="drawing" width="450"/>
</p>

For the rectangle and trapezoidal rule m is equals 2 and for Simpson's rule m is equals 4. After substituting this formula into the integral equation and neglecting the small error of the quadrature formula, for each partition point of the segment [a,b] we obtain a linear equation:

<p align="center">
  <img src="img/system.png" alt="drawing" width="450"/>
</p>

Finally, solving this system, we obtain an approximation for the desired function by the formula:

<p align="center">
  <img src="img/approximation.png" alt="drawing" width="450"/>
</p>

The mesh convergence is estimated by the relation:

<p align="center">
  <img src="img/condition.png" alt="drawing" width="200"/>
</p>

where the integral in the [norm of the L2 space](https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm) is calculated using [adaptive quadrature](https://en.wikipedia.org/wiki/Adaptive_quadrature)

# Testing
<p align="center">
  <img src="img/test1.png" alt="drawing" width="800"/>
  <img src="img/test2.png" alt="drawing" width="800"/>
</p>

# Dependencies

CUDA, cuBLAS, cuSOLVER, gnuplot
