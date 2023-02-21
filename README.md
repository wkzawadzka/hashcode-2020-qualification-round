# hashcode-2020-qualification-round
## About
A solver for 2020 Google's Hashcode qualification round using greedy & genetic approaches.

## Problem
We divided the problem into 4 main questions:
- Which libraries to choose?
- What is the best permutation of chosen libraries?
- How many books to choose from each library?
- What is the best permutation of those books?

## Greedy
Elimination of libraries

 ```sum([self.book_scores[x] for x in library.books[:(self.days-library.signup)*library.ships_per_day]])**importance)/library.signup```
 
The equation above let us set a score for each library. It is based on the highest score of books scanned in the available time if said library would be picked as first to the power of importance if this score, divided by the time it takes for the library to sign up. The importance factor is changing based on the instance and was set based on experiments.

The algorithm went through every library, added to the overall score the highest value books that it could fit in the given time, and added these books to the set of books already scanned.
 
## Genetic
We answer the question of how many books to take as the maximum books possible to scan in a given deadline having already a permutation of libraries (chosen greedily). Mutation here was performed by taking a random library in a given solution and changing its books’ order and/or selection. Tournament selection was used to choose the population at each step. If there was no improvement in highest fitness after 5 iterations, the genetic algorithm stopped. One-point crossover was used. Each parent gave a portion of their solution for given libraries to the child. 


**Achieved total score: 26 585 136.**
