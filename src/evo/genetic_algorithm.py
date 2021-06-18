import datetime
import os
import pickle

import numpy as np
from tqdm import tqdm


class GeneticOptimizer:
    def __init__(self, specimen_count, survivor_count, max_iterations, mutation_rate,
                 generation_method="Random Splice", fitness_target=None, output_dir=None):
        self.specimen_count = specimen_count  # total number of specimenCount to compete
        self.survivor_count = survivor_count  # number of specimenCount to survive each generation
        self.children_count = int(
            self.specimen_count / self.survivor_count)  # calculated number of children for each survivor
        self.wild_children_count = np.max(
            [int(self.children_count / 10), 1])  # Wild mutation rate - set to 10% but not less than 1
        self.max_iterations = max_iterations  #
        self.base_mutation_rate = mutation_rate  # rate of change with each new generation
        self.generation_method = generation_method  # Random Splice/ Pure Mutation

        self.fitness_target = fitness_target

        self.fitness_history = []
        self.best_survivor_history = []
        self.best_survivor = None
        self.output_dir = output_dir

    def get_best_specimens(self, generation, fitness_function):
        # Calculate fitness of current generation
        fitness = np.zeros([self.specimen_count, 1])
        for i in range(self.specimen_count):
            fitness[i, 0] = fitness_function(generation[i])
        # Sort - HIGHEST fitness first
        ind = np.argsort(fitness, axis=0)[:self.survivor_count].squeeze()
        # ind = np.flip(ind[-self.survivor_count:].squeeze())
        # Save fitness of the best survivors and calculate the change in fitness from the previous generation
        best_fit = fitness[ind[0], 0]
        best_genes = [generation[i] for i in ind]  # Surviving survivorCount specimens
        return best_genes, best_fit

    @staticmethod
    def calculate_gene_var(best_parents):
        # Calculate the total mean of each variable (out of the survivors)
        number_of_samples = len(best_parents)
        gene_var = []
        for gene in best_parents[0]:
            gene_var.append([0] * len(gene))

        for parent in best_parents:
            for i, gene in enumerate(parent):
                if type(gene) == list:
                    if len(gene) == 0:
                        continue
                    else:
                        for j, subvar in enumerate(gene):
                            gene_var[i][j] += np.std(subvar) / number_of_samples
                else:
                    gene_var[i] += np.std(gene) / number_of_samples

        return gene_var

    @staticmethod
    def _base_mutation(gene, mutation_rate):
        mutated_gene = gene + mutation_rate * np.mean(gene) * (
                np.random.random_sample(gene.shape) - 0.5)
        return mutated_gene

    @staticmethod
    def _spliced_mutation(gene, splicing_gene, mutation_rate):
        mask = np.round(np.random.random_sample(gene.shape))
        mutated_gene = mask * gene + (1 - mask) * splicing_gene
        mutated_gene += mutation_rate * np.mean(gene) * (
                np.random.random_sample(gene.shape) - 0.5)
        return mutated_gene

    def breed_new_generation(self, best_parents):
        # This function takes the best survivors and breeds a new generation
        gene_var = self.calculate_gene_var(best_parents)
        # Generate new generation from child of survivors
        new_generation = []
        for s in range(self.survivor_count):
            current_parent = best_parents[s]
            for c in range(self.children_count):
                if c >= self.children_count - self.wild_children_count:
                    wild_mutation = (np.random.rand() + 0.1) * 10
                else:
                    wild_mutation = 1
                mutation_rate = wild_mutation * self.base_mutation_rate
                child = list()
                if self.generation_method == "Pure Mutation":
                    for i, gene in enumerate(current_parent):
                        if type(gene) is list:
                            mutated_gene = list()
                            for j, sub_gene in enumerate(gene):
                                mutated_gene.append(self._base_mutation(sub_gene, gene_var[i][j] * mutation_rate))
                        else:
                            mutated_gene = self._base_mutation(gene, gene_var[i] * mutation_rate)
                        child.append(mutated_gene)
                elif self.generation_method == "Random Splice":
                    spliced = np.random.randint(len(best_parents))
                    splicing_partner = best_parents[spliced]
                    for i, gene in enumerate(current_parent):
                        splicing_gene = splicing_partner[i]
                        mutated_gene = list()
                        if type(gene) is list:
                            for j, sub_gene in enumerate(gene):
                                mutated_gene.append(
                                    self._spliced_mutation(sub_gene, splicing_gene[j], gene_var[i][j] * mutation_rate))
                        else:
                            mutated_gene = self._spliced_mutation(gene, splicing_gene, gene_var[i] * mutation_rate)
                        child.append(mutated_gene)
                else:
                    raise Exception(f'generation method must be Pure Mutation/Random Splice')
                new_generation.append(child)
            # Add the two best survivors of the previous generation (this way the best fitness never goes down)
            new_generation[:2] = best_parents[:2]
        return new_generation

    def initialize_generation(self, variable_list):
        generation = list()
        for i in range(self.specimen_count):
            child = list()
            for var in variable_list:
                if type(var) == list:
                    child.append([np.random.random_sample(subvar.shape) - 0.5 for subvar in var])
                else:
                    child.append(np.random.random_sample(var.shape) - 0.5)
            generation.append(child)
        return generation

    def optimize(self, variable_list, fitness_function):
        """

        :param variable_list:  list of arrays with the shapes of the optimizable variables,
        # which should also be the input to fitnessFunction
        :param fitness_function: function type, calculates the fitness of current generation
        :return:
        """

        if self.output_dir is not None:
            FORMAT = "%Y_%m_%d-%H_%M"
            ts = datetime.datetime.now().strftime(FORMAT)
            run_name = fitness_function.name + '_' + ts
            self.output_dir = os.path.join(self.output_dir, run_name)
            os.makedirs(self.output_dir, exist_ok=True)
            print(f'parameters will be saved to {self.output_dir}')

        generation = self.initialize_generation(variable_list)
        self.fitness_history = []
        self.best_survivor_history = []
        pbar = tqdm(range(self.max_iterations))
        itr = 0
        best_fit = None
        for itr in pbar:
            best_genes, best_fit = self.get_best_specimens(generation, fitness_function)
            generation = self.breed_new_generation(best_genes)

            pbar.desc = f'Best fitness: {best_fit:.2f}'
            self.best_survivor = generation[0]
            self.fitness_history.append(best_fit)
            self.best_survivor_history.append(self.best_survivor)

            if self.output_dir is not None:
                fitness_str = str(f'{best_fit:.2f}'.replace('.', '_'))
                agent_name = f'agent_parameters_{itr}_fitness_{fitness_str}.pkl'

                full_output_path = os.path.join(self.output_dir, agent_name)
                with open(full_output_path, 'wb') as file:
                    pickle.dump(self.best_survivor, file)

            if self.fitness_target is not None:
                if best_fit < self.fitness_target:
                    print(f"breaking: current best fitness {best_fit} under target {self.fitness_target}")
                    break
        print('Last Iteration: {}, Best fitness: {}'.format(itr, best_fit))