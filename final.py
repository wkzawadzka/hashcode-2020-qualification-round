import time
import numpy as np
from typing import List
import random
import sys

''' 
********************************************************

    Solver for Qualification Round of Google Hashcode 2020 
    
********************************************************

@version
Code is written in Python 3.9.6.

@authors
wkzawadzka 
elissa-c

@dependencies
numpy 1.23.3
typing 3.10.0.0

@usage
Program takes standard input and writes to standard output. 
One argument, instance type, is needed.

Examplar usage
In PowerShell 7.3.1:
    get-Content {path to instance} | python final.py {instance type}
    get-Content ./input/a.txt | python final.py a

    redirecting to file:
    get-Content ./input/a.txt | python final.py a > a_out.txt
In CMD:
    python final.py {instance type} < {path to instance}
    python final.py a < ./input/a.txt

    redirecting to file:
    python final.py a < ./input/a.txt > a_out.txt

'''


class Library:
    def __init__(self, id: int, library_info: dict, books: np.array):
        self.id = id
        self.num_books = library_info['books']
        self.signup = library_info['signup_time']
        self.ships_per_day = library_info['ships_per_day']
        self.books = books  # np.array
        self.max_books_possible = 0  # needed for later


class Problem:
    def __init__(self, greedy_importance_factor: int):
        self.importance = greedy_importance_factor
        self.num_books = 0
        self.num_libraries = 0
        self.books_scores = np.array
        self.days = 0
        self.libraries = []
        self.read_input()

    def read_input(self):
        # list of dictionaries {'books':n, 'signup_time':t, 'ships_per_day':m}
        libraries_info = []
        libraries_books = []  # list of np arrays with sorted by value books
        for i, line in enumerate(sys.stdin):
            if line.strip():
                if i == 0:
                    self.num_books, self.num_libraries, self.days = map(
                        int, line.split())
                elif i == 1:
                    self.book_scores = np.array(list(map(int, line.split())))
                else:
                    if i % 2 == 0:
                        n, t, m = map(int, line.split())
                        libraries_info.append(
                            {'books': n, 'signup_time': t, 'ships_per_day': m})
                    else:
                        # sorted by better first
                        lib_books = np.array(list(map(int, line.split())))
                        lib_scores = np.array(
                            [self.book_scores[book] for book in lib_books])
                        inds = (-lib_scores).argsort()  # decreasing order
                        libraries_books.append(lib_books[inds])
        self.create_libraries(libraries_info, libraries_books)
        self.limit_libraries(self.importance)

    def create_libraries(self, libraries_info: dict, libraries_books: np.array):
        for i in range(self.num_libraries):
            self.libraries.append(
                Library(i, libraries_info[i], libraries_books[i]))

    def limit_libraries(self, importance: int):
        libs = []

        for library in self.libraries:
            libs.append([(sum([self.book_scores[x] for x in library.books[:(
                self.days-library.signup)*library.ships_per_day]])**importance)/library.signup, library])

        libs.sort(key=lambda x: x[0], reverse=True)
        libs = [x[1] for x in libs]

        day = 0
        i = 0
        for library in libs:
            if (day + library.signup >= self.days):
                break
            library.max_books_possible = min(
                library.num_books,  (self.days - day - library.signup)*library.ships_per_day)
            day += library.signup
            i += 1
        libs = libs[:i]

        # update information
        self.libraries = libs
        self.num_libraries = len(libs)
        self.total_signup_days = sum(
            self.libraries[i].signup for i in range(self.num_libraries))


class GeneticSolver:
    def __init__(self, problem: Problem, population_size: int, tournament_size: int, mutation_probability: float, mating_pool_size: int) -> None:
        self.result = self.evolutionary_approach(problem.book_scores, problem.libraries,
                                                 population_size, tournament_size, mutation_probability, mating_pool_size)

    def evolutionary_approach(self, bookscores: np.array, libraries: List[Library], pop_size: int, k: int, p: float, mating_pool_size: int):
        no_improvement = 0
        pop = self.create_population(pop_size, libraries)
        population = self.evaluate_population(bookscores, pop)
        #assert(population[0][1] >= population[1][1])

        global_max = population[0][1]
        solution = population[0][0]
        generation = 0
        while True:
            #print(f'for generation {generation+1} best f: {global_max}')
            #assert(population[0][1] >= population[pop_size-1][1])

            mating_pool = self.mating_pool(mating_pool_size, k, population)

            # get children
            children = self.get_children(bookscores, libraries, mating_pool, p)
            best_of_children = self.get_maximum(children)
            local_max, local_sol = best_of_children[1], best_of_children[0]

            if local_max > global_max:
                no_improvement = 0
                global_max = local_max
                solution = local_sol
            else:
                no_improvement += 1
            population.extend(children)
            population = sorted(population, key=lambda x: x[1], reverse=True)

            # remove worst with tournament
            best = [local_sol]
            for _ in range(pop_size-1):
                best.append(self.tournament_selection(k, population))
            population.clear()
            population = self.evaluate_population(bookscores, best)
            #assert(population[0][1] >= population[-1][1])
            generation += 1

            if no_improvement == 5 or generation == 100:
                break

        #print(f'BEST FOUND: {global_max} for solution {solution}')
        return (solution, global_max)

    def is_valid(self, libraries: List[Library], solution: List[np.array]):
        #assert len(solution) == len(libraries)
        for i, libSol in enumerate(solution):
            if len(set(libSol)) != len(libSol):
                return False
            if len(set(libSol)) > libraries[i].max_books_possible:
                return False
            if (set(libSol)-set(libraries[i].books) != set()):
                return False
        return True

    def evaluate_solution(self, bookscores: np.array, solution: List[np.array]):
        # fitness = book scores
        books = list(set(int(i) for sub in solution for i in sub))
        books = np.array(books)
        return sum(bookscores[books])

    def evaluate_population(self, bookscores: np.array, population: List[List[np.array]]):
        pop_evaluated = []
        for sol in population:
            ev = self.evaluate_solution(bookscores, sol)
            pop_evaluated.append([sol, ev])
        # highest fitness first
        pop_evaluated = sorted(pop_evaluated, key=lambda x: x[1], reverse=True)
        #assert(pop_evaluated[0][1] >= pop_evaluated[1][1])
        return pop_evaluated

    def create_population(self, pop_size: int, libraries: List[Library]):
        pop = []
        while len(pop) < pop_size:
            solution = [np.random.choice(
                a=libraries[i].books, size=libraries[i].max_books_possible, replace=False) for i in range(len(libraries))]
            if self.is_valid(libraries, solution):
                pop.append(solution)
        return pop

    def get_maximum(self, population: List[List]):
        a = sorted(population, key=lambda x: x[1], reverse=True)
        #assert(a[0][1] >= a[1][1])
        return a[0]

    def tournament_selection(self, k: int, population: List[List[np.array]]):
        random_choices = random.choices(population, k=k)
        best = self.get_maximum(random_choices)
        return best[0]

    def mating_pool(self, size: int, k: int, population: List[List[np.array]]):
        mating_pool = []
        for _ in range(size):
            parents = []
            for _ in range(2):
                parents.append(self.tournament_selection(k, population))
            mating_pool.append(parents)
        return mating_pool

    def mutate(self, sol: List[np.array], p: float, libraries: List[Library]):
        if np.random.random() > p:
            for _ in range(int(len(libraries)/5)):
                i = random.randint(0, len(sol)-1)
                sol[i] = np.random.choice(
                    a=libraries[i].books, size=libraries[i].max_books_possible, replace=False)
        return sol

    def crossover(self, sol1: List[np.array], sol2: List[np.array]):
        child = []
        i = random.randint(1, len(sol1)-1)
        child.extend(sol1[:i])
        child.extend(sol2[i:])
        return child

    def get_children(self, bookscores, libraries, mating_pool, p):
        children = []
        for pair in mating_pool:
            child = self.crossover(pair[0], pair[1])
            child = self.mutate(child, p, libraries)
            f = self.evaluate_solution(bookscores, child)
            children.append([child, f])
        return children


class GreedySolver:
    def __init__(self, problem: Problem) -> None:
        self.result = self.perform(problem)

    def perform(self, problem: Problem):
        solution = []
        donebooks = set()
        sco = 0
        days_left = problem.days
        for lib in problem.libraries:
            sol = set()
            days_left -= lib.signup
            val = scanned = itter = 0
            while itter < lib.num_books and scanned < (days_left)*lib.ships_per_day:
                book = lib.books[itter]
                if book not in donebooks:
                    val += problem.book_scores[book]
                    donebooks.add(book)
                    scanned += 1
                    sol.add(book)
                itter += 1
            sco += val
            solution.append(np.array(list(sol)))
        return (solution, sco)


def get_output(solution, problem):
    sys.stdout.write(f"{len(problem.libraries)}\n")
    for i, library in enumerate(problem.libraries):
        sys.stdout.write(f"{library.id} {len(solution[i])}\n")
        sys.stdout.write(" ".join(str(x) for x in solution[i])+"\n")


if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) == 1:
        print(
            "Provide argument of instance type.\nUsage: python final.py {instance type} < {path to instance}")
        exit()

    instance = sys.argv[1]

    # parameters
    greedy_importance_factor = 1
    mutation_probability = 0.3
    tournament_size = 3
    population_size = 50
    mating_pool_size = 0

    if instance == "a":
        pass

    elif instance == "b":
        mating_pool_size = 5

    elif instance == "c":
        population_size = 100
        tournament_size = 20
        mating_pool_size = 30

    elif instance == "d":
        mating_pool_size = 5

    elif instance == "e":
        greedy_importance_factor = 1.08
        population_size = 250
        mutation_probability = 0.5
        mating_pool_size = 50

    elif instance == "f":
        greedy_importance_factor = 0.81
        population_size = 500
        tournament_size = 50
        mating_pool_size = 10

    else:
        print("Wrong instance. Try again")
        exit()

    problem = Problem(greedy_importance_factor)

    # basic greedy algorithm
    greedy_solver = GreedySolver(problem)
    greedy_solution, greedy_points = greedy_solver.result

    if (instance == "a"):
        get_output(greedy_solution, problem)
        stop_time = time.time()
        exit()

    # genetic algorithm
    genetic_solver = GeneticSolver(
        problem, population_size, tournament_size, mutation_probability, mating_pool_size)
    genetic_solution, genetic_points = genetic_solver.result

    # choose solution giving higher number of points
    solution = greedy_solution
    if genetic_points > greedy_points:
        solution = genetic_solution

    # output
    get_output(solution, problem)
    stop_time = time.time()
    #print(f"time taken: {stop_time-start_time}")
