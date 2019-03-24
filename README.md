Squaring numbers:
https://www.youtube.com/watch?v=yNs8B1VnMAA&index=41&list=PLGvfHSgImk4aweyWlhBXNF6XISY3um82_

## Writing efficient programs: 

### High-level stratergies
More than 3 Trillion * 3 Tera-FLOPS per second
System fetches operand from memory => time waste
=> 
1. Maximise arithmetic intensity = (amount of math done) / (amount of memory accessed)

    - Maximise number of usefule compute ops (work) per thread (numerator)
    - Minimise time spent on memory accesses per thread (denomenator)
        * move frequently-accessed data to fast memory

        local (registers or L1-cache) >  shared  >>  global >> host (CPU)
