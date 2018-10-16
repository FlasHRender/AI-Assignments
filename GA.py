import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt


class DiffEvolGA():
    # ********************************************************************************************* #
    # Population can be initialized externally or internally
    _population = None
    crossOverProb = None
    popSize = None
    noOfGens = None
    # If change of best solution is below this threshold, sim is stopped
    threshold = None
    rel_k = None
    rel_f = None
    func = None
    vecDimen = None
    _debug = None
    _ZombiePopulation = None
    _trailPop = None
    _pltHandle = None
    # ********************************************************************************************** #

    def __init__(self, func, low, high, vecDimen=2, popSize=300, population=None, noOfGens=150, crossOverProb=0.4, threshold=10**-10, rel_k=0.5, rel_f=-1, debug=False, psave=None):
        
        # Initialize random population
        if (population is None):            
            popTemp = (np.array(high, dtype=float) - np.array(low, dtype=float)) * \
                np.random.random_sample([popSize, vecDimen]) + np.array(low)
            # val = (b-a)*rand + a # Scaling values
            
            self._population = np.concatenate(
                (popTemp, np.Inf + np.zeros([popSize, 1])), axis=1)
        
        else:
            if (population.shape != (popSize, vecDimen)):
                raise ValueError(
                    "Error: Population should be a numpy array with shape", (popSize, vecDimen))
            else:
                self._population = np.concatenate(
                    (population, np.Inf + np.zeros([popSize,])), axis=1)
                # +1 dimen to store fitness of each point

        assert sum([x<y for x, y in zip(low, high)]) == vecDimen, "Check limits" 
        self.lim = [low, high]
        self.vecDimen = vecDimen
        self.popSize = popSize
        self.crossOverProb = crossOverProb
        self.noOfGens = noOfGens
        self.func = func
        self.threshold = threshold
        self.rel_f = rel_f
        self.rel_k = rel_k
        self.psve = psave
        self._debug = debug
        self._ZombiePopulation = np.zeros(self._population.shape)
        self._trailPop = np.zeros(self._population.shape)
        self._random_array = np.zeros(self._population.shape)
        self._mask = np.full((popSize), True)

    def _newPopulation(self):
        self._mutate()      # Zombies updated
        self._genTrailVec() # Crossed over, updated Zombie points
        self._checkLimits() # Reset points into limits

        # Calculate fitness of all trail-vector points
        for xombie in self._trailPop:
            xombie[-1] = self.func(xombie[:-1])

        for indv in self._population:
            indv[-1] = self.func(indv[:-1])
        # Now, trail vectors have their fitnesses, now enforce elitism
        self._elite()
        # new population is done, return

    def _mutate(self):
        for ix in range(0, self.popSize):
            r1, r2, r3 = np.random.randint(0, self.popSize, 3)
            zombie = self._population[ix] + self.rel_k*(self._population[r1] - self._population[ix]) + self.rel_f*(self._population[r2] - self._population[r3])
            # x = x + k(r1 - x) + F*(r2 - r3)
            # can the above expression be done without loops?
            self._ZombiePopulation[ix] = zombie
        # Mutation done, caller now supposed to generate trail vectors using crossover

    def _genTrailVec(self):
        # Using nice numpy array manipulation, we can express
        # for each coord in each pop member
        #   if rand < crossOverProb,
        #      use the mutant coord
        #   else
        #      use the parent point
        # AS,
        self.random_array = np.random.random_sample([self.popSize, self.vecDimen + 1])
        
        self._trailPop = (self._random_array < self.crossOverProb)*self._ZombiePopulation + \
            (self._random_array > self.crossOverProb)*self._population
        
        # At this stage, population hasn't been modified yet, so consistent.
        # Exit, and check fitness of the trail population.

    
    def _checkLimits(self, sw=False):
        if (not sw):
            # Get mask where points are larger
            # np.bitwise_or.reduce( (self._trailPop[:,:-1] > self.lim[0])*(self._trailPop[:,:-1] < self.lim[1]) , axis=1, out=self._mask)
            # indx = np.where(self._mask == True)[0]
            
            for i in range(0, self.popSize):
                self._trailPop[i,:-1] = self._trailPop[i,:-1]%[h-l for h, l in zip(self.lim[1], self.lim[0])] + self.lim[0]

        else:
            # np.bitwise_or.reduce( (self._population[:,:-1] > self.lim[0])*(self._population[:,:-1] < self.lim[1]) , axis=1, out=self._mask)
            # indx = np.where(self._mask == True)[0]
            
            for i in range(self.popSize):
                self._population[i,:-1] = self._population[i,:-1]%[h-l for h, l in zip(self.lim[1], self.lim[0])] + self.lim[0]

            for indv in self._population:
                indv[-1] = self.func(indv[:-1])

    def _elite(self):
        # Basically,
        # if fit(xombie) > fit(popmember)
        #   popmember = xombie
        # # else do nothing.
        # >> a[:-1,:-1]
        popSz = self.popSize                                                                        # below, pop is fitter, so retain pop
        self._population = \
            (self._population[:, -1] < self._trailPop[:, -1]).reshape(popSz, 1)*self._population + \
            (self._population[:, -1] > self._trailPop[:, -1]).reshape(popSz, 1)*self._trailPop     # replace with trail pop as it is fitter

    def dumpPop(self, gen=None):
        if (gen is not None):
            print("\n\nGEN", gen)
        print(self._population)

    def simulate(self):
        # First generation is same as initial population.
        currentGen = 0
        currentThresh =  np.Inf
        minAll =         np.Inf
        maxAll =       -1*np.Inf
        # One time initial evaluation of population
        for indv in self._population:
            indv[-1] = self.func(indv[:-1])

        self._checkLimits(True)

        avgArr = []
        minArr = []
        maxArr = []

        while (currentGen < self.noOfGens and self.threshold < currentThresh):

            # Mutates, Crosses-over and implements Elitism, modifying the population in place.
            self._newPopulation()
            currentGen += 1
            self.rel_k = 4*np.random.random_sample() - 2
            # self.dumpPop(currentGen)

            minGen = np.min(self._population[:, -1])
            maxGen = np.max(self._population[:, -1])
            avgGen = np.average(self._population[:, -1])

            currentThresh = abs(avgGen - maxGen)
            
            # Sanity Check, assert True => nothing happens
            # assert (minAll >= minGen), "WARNING! Elitism condition failing!"
            
            minAll = np.min([minGen, minAll])
            maxAll = np.max([maxAll, maxGen])

            avgArr.append(avgGen)
            minArr.append(minGen)
            maxArr.append(maxGen)
            
            if (self._debug):
                print("Gen:", currentGen, "Min: ", str.format("{0:.4e}",minGen), "Max", maxGen)
                # self.dumpPop()
            # After this step, plot and go to next generation

        bestInd = np.argmin(self._population[:,-1])
        # print("\n>>>", bestInd, self._population[bestInd])
        best = self._population[bestInd][-1]
        point = self._population[bestInd][:-1]

        if (currentGen == self.noOfGens):
            print("Max Generations (" + str(self.noOfGens) + ") reached. \n\tBest", str.format("{0:.4e}",best), "Point: ", point)
        else:
            print("Min threshold reached. \n\tBest", str.format("{0:.4e}", best), "Point: ", point)

        plt.clf()
        # 1162, 832
        plt.figure(figsize=(11.62, 8.32), dpi=100)
        plt.plot([x for x in range(currentGen)], minArr, 'b.', label='Minimum')
        plt.plot([x for x in range(currentGen)], avgArr, '-', label='Average')
        plt.plot([x for x in range(currentGen)], maxArr, 'r.', label='Maximun')
        plt.xlim([0, self.noOfGens])
        plt.ylim([min(minArr) - 0.1*(max(avgArr) - min(minArr)), max(avgArr) ])
        plt.legend(loc='center left')
        plt.grid()
        #plt.show()
        plt.savefig(self.psve + " Fit[" + str.format("{0: .04}",best) + "] Pnt" + str(point) + ".png", bbox_inches='tight', )

if __name__ == "__main__":
    
    def Egg(coord):
        x = coord[0]
        y = coord[1]
        return -1*(y+47) * (math.sin(math.sqrt(abs(x*0.5 + y + 47)))) - x*math.sin(abs(x-y-47))

    def Tab(coord):
        x = coord[0]
        y = coord[1]
        return -1*abs(math.sin(x)*math.cos(y)*math.exp(abs(1-(math.sqrt(x**2 + y**2))/math.pi)))

    p = [20, 50, 100, 200]
    g = [50, 100, 200]
    
    f = [Egg, Tab]
    sims = []
    m = [[[-512, -512],[512, 512]], [[-10, -10],[10, 10]]]
    for func, lims in zip(f, m):
        for gen in g:
            for pop in p:
                pltsave = func.__name__ + " Gen-" + str(gen) + " Pop-" + str(pop)
                sims.append(DiffEvolGA(func, lims[0], lims[1], noOfGens=gen, popSize=pop, crossOverProb=0.8, psave=pltsave ))
                sims[-1].simulate()
