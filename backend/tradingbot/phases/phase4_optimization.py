"""
Phase 4: Strategy Optimization Engine
Parameter tuning and strategy optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random

from ..core.production_logging import ProductionLogger
from ..core.production_config import ConfigManager
from .phase4_backtesting import BacktestEngine, BacktestConfig, BacktestResults


class OptimizationMethod(Enum): 
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"


class OptimizationMetric(Enum): 
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class ParameterRange: 
    """Parameter range for optimization"""
    name: str
    min_value: float
    max_value: float
    step: float
    param_type: str  # "int", "float", "bool"


@dataclass
class OptimizationConfig: 
    """Optimization configuration"""
    method: OptimizationMethod
    metric: OptimizationMetric
    max_iterations: int
    parameter_ranges: List[ParameterRange]
    backtest_config: BacktestConfig
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5


@dataclass
class OptimizationResult: 
    """Optimization result"""
    best_parameters: Dict[str, Any]
    best_score: float
    best_results: BacktestResults
    all_results: List[Tuple[Dict[str, Any], float, BacktestResults]]
    optimization_time: float
    iterations_completed: int


class StrategyOptimizer: 
    """Strategy optimization engine"""
    
    def __init__(self, 
                 backtest_engine: BacktestEngine,
                 config: ConfigManager,
                 logger: ProductionLogger):
        self.backtest_engine = backtest_engine
        self.config = config
        self.logger = logger
        
        self.logger.info("StrategyOptimizer initialized")
    
    async def optimize_strategy(self, strategy, optimization_config: OptimizationConfig)->OptimizationResult:
        """Optimize strategy parameters"""
        try: 
            self.logger.info(f"Starting optimization using {optimization_config.method.value}")
            
            start_time = datetime.now()
            all_results = []
            
            if optimization_config.method  ==  OptimizationMethod.GRID_SEARCH: 
                all_results = await self._grid_search(strategy, optimization_config)
            elif optimization_config.method ==  OptimizationMethod.RANDOM_SEARCH: 
                all_results = await self._random_search(strategy, optimization_config)
            elif optimization_config.method ==  OptimizationMethod.GENETIC_ALGORITHM: 
                all_results = await self._genetic_algorithm(strategy, optimization_config)
            else: 
                raise ValueError(f"Unsupported optimization method: {optimization_config.method}")
            
            # Find best result
            best_result = max(all_results, key = lambda x: x[1])
            best_parameters, best_score, best_backtest_results = best_result
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                best_parameters = best_parameters,
                best_score = best_score,
                best_results = best_backtest_results,
                all_results = all_results,
                optimization_time = optimization_time,
                iterations_completed = len(all_results)
            )
            
            self.logger.info(f"Optimization completed: {len(all_results)} iterations, best score: {best_score:.4f}")
            return result
            
        except Exception as e: 
            self.logger.error(f"Error optimizing strategy: {e}")
            raise
    
    async def _grid_search(self, strategy, config: OptimizationConfig)->List[Tuple[Dict[str, Any], float, BacktestResults]]: 
        """Grid search optimization"""
        results = []
        
        # Generate all parameter combinations
        parameter_combinations = self._generate_parameter_combinations(config.parameter_ranges)
        
        # Limit combinations if too many
        if len(parameter_combinations)  >  config.max_iterations: 
            parameter_combinations = parameter_combinations[: config.max_iterations]
        
        for i, params in enumerate(parameter_combinations): 
            try: 
                self.logger.info(f"Grid search iteration {i + 1}/{len(parameter_combinations)}")
                
                # Update strategy parameters
                await self._update_strategy_parameters(strategy, params)
                
                # Run backtest
                backtest_results = await self.backtest_engine.run_backtest(strategy, config.backtest_config)
                
                # Calculate score
                score = self._calculate_score(backtest_results, config.metric)
                
                results.append((params, score, backtest_results))
                
            except Exception as e: 
                self.logger.error(f"Error in grid search iteration {i + 1}: {e}")
                continue
        
        return results
    
    async def _random_search(self, strategy, config: OptimizationConfig)->List[Tuple[Dict[str, Any], float, BacktestResults]]: 
        """Random search optimization"""
        results = []
        
        for i in range(config.max_iterations): 
            try: 
                self.logger.info(f"Random search iteration {i + 1}/{config.max_iterations}")
                
                # Generate random parameters
                params = self._generate_random_parameters(config.parameter_ranges)
                
                # Update strategy parameters
                await self._update_strategy_parameters(strategy, params)
                
                # Run backtest
                backtest_results = await self.backtest_engine.run_backtest(strategy, config.backtest_config)
                
                # Calculate score
                score = self._calculate_score(backtest_results, config.metric)
                
                results.append((params, score, backtest_results))
                
            except Exception as e: 
                self.logger.error(f"Error in random search iteration {i + 1}: {e}")
                continue
        
        return results
    
    async def _genetic_algorithm(self, strategy, config: OptimizationConfig)->List[Tuple[Dict[str, Any], float, BacktestResults]]: 
        """Genetic algorithm optimization"""
        results = []
        
        # Initialize population
        population = self._initialize_population(config.parameter_ranges, config.population_size)
        
        for generation in range(config.max_iterations // config.population_size): 
            try: 
                self.logger.info(f"Genetic algorithm generation {generation + 1}")
                
                generation_results = []
                
                # Evaluate population
                for individual in population: 
                    params = self._individual_to_parameters(individual, config.parameter_ranges)
                    
                    # Update strategy parameters
                    await self._update_strategy_parameters(strategy, params)
                    
                    # Run backtest
                    backtest_results = await self.backtest_engine.run_backtest(strategy, config.backtest_config)
                    
                    # Calculate score
                    score = self._calculate_score(backtest_results, config.metric)
                    
                    generation_results.append((individual, score, backtest_results))
                    results.append((params, score, backtest_results))
                
                # Sort by score
                generation_results.sort(key = lambda x: x[1], reverse = True)
                
                # Select elite
                elite = [individual for individual, score, _ in generation_results[: config.elite_size]]
                
                # Generate new population
                new_population = elite.copy()
                
                while len(new_population)  <  config.population_size: 
                    # Selection
                    parent1 = self._tournament_selection(generation_results)
                    parent2 = self._tournament_selection(generation_results)
                    
                    # Crossover
                    if random.random()  <  config.crossover_rate: 
                        child1, child2 = self._crossover(parent1, parent2, config.parameter_ranges)
                        new_population.extend([child1, child2])
                    else: 
                        new_population.extend([parent1, parent2])
                
                # Mutation
                for i in range(len(new_population)): 
                    if random.random()  <  config.mutation_rate: 
                        new_population[i] = self._mutate(new_population[i], config.parameter_ranges)
                
                population = new_population[: config.population_size]
                
            except Exception as e: 
                self.logger.error(f"Error in genetic algorithm generation {generation + 1}: {e}")
                continue
        
        return results
    
    def _generate_parameter_combinations(self, parameter_ranges: List[ParameterRange])->List[Dict[str, Any]]: 
        """Generate all parameter combinations for grid search"""
        combinations = [{}]
        
        for param_range in parameter_ranges: 
            new_combinations = []
            
            if param_range.param_type  ==  "int": values = range(int(param_range.min_value), int(param_range.max_value) + 1, int(param_range.step))
            elif param_range.param_type ==  "float": values = [param_range.min_value + i * param_range.step 
                         for i in range(int((param_range.max_value - param_range.min_value) / param_range.step) + 1)]
            elif param_range.param_type ==  "bool": values = [True, False]
            else: 
                values = [param_range.min_value]
            
            for combination in combinations: 
                for value in values: 
                    new_combination = combination.copy()
                    new_combination[param_range.name] = value
                    new_combinations.append(new_combination)
            
            combinations = new_combinations
        
        return combinations
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange])->Dict[str, Any]: 
        """Generate random parameters"""
        params = {}
        
        for param_range in parameter_ranges: 
            if param_range.param_type  ==  "int": params[param_range.name] = random.randint(int(param_range.min_value), int(param_range.max_value))
            elif param_range.param_type ==  "float": params[param_range.name] = random.uniform(param_range.min_value, param_range.max_value)
            elif param_range.param_type ==  "bool": params[param_range.name] = random.choice([True, False])
            else: 
                params[param_range.name] = param_range.min_value
        
        return params
    
    def _initialize_population(self, parameter_ranges: List[ParameterRange], population_size: int)->List[List[float]]:
        """Initialize population for genetic algorithm"""
        population = []
        
        for _ in range(population_size): 
            individual = []
            for param_range in parameter_ranges: 
                if param_range.param_type  ==  "int": individual.append(random.randint(int(param_range.min_value), int(param_range.max_value)))
                elif param_range.param_type ==  "float": individual.append(random.uniform(param_range.min_value, param_range.max_value))
                elif param_range.param_type ==  "bool": individual.append(random.choice([0.0, 1.0]))
                else: 
                    individual.append(param_range.min_value)
            
            population.append(individual)
        
        return population
    
    def _individual_to_parameters(self, individual: List[float], parameter_ranges: List[ParameterRange])->Dict[str, Any]: 
        """Convert individual to parameters"""
        params = {}
        
        for i, param_range in enumerate(parameter_ranges): 
            value = individual[i]
            
            if param_range.param_type  ==  "int": params[param_range.name] = int(value)
            elif param_range.param_type ==  "float": params[param_range.name] = value
            elif param_range.param_type  ==  "bool": params[param_range.name] = bool(value)
            else: 
                params[param_range.name] = value
        
        return params
    
    def _tournament_selection(self, population_results: List[Tuple[List[float], float, BacktestResults]], 
                            tournament_size: int = 3)->List[float]:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population_results, min(tournament_size, len(population_results)))
        winner = max(tournament, key = lambda x: x[1])
        return winner[0]
    
    def _crossover(self, parent1: List[float], parent2: List[float], 
                   parameter_ranges: List[ParameterRange])->Tuple[List[float], List[float]]: 
        """Crossover operation for genetic algorithm"""
        child1 = []
        child2 = []
        
        for i in range(len(parent1)): 
            if random.random()  <  0.5: 
                child1.append(parent1[i])
                child2.append(parent2[i])
            else: 
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def _mutate(self, individual: List[float], parameter_ranges: List[ParameterRange])->List[float]:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        
        for i, param_range in enumerate(parameter_ranges): 
            if random.random()  <  0.1:  # 10% mutation rate per parameter
                if param_range.param_type ==  "int": mutated[i] = random.randint(int(param_range.min_value), int(param_range.max_value))
                elif param_range.param_type ==  "float": mutated[i] = random.uniform(param_range.min_value, param_range.max_value)
                elif param_range.param_type ==  "bool": mutated[i] = random.choice([0.0, 1.0])
        
        return mutated
    
    async def _update_strategy_parameters(self, strategy, parameters: Dict[str, Any]): 
        """Update strategy parameters"""
        try: 
            # Mock parameter update - in production, update actual strategy parameters
            for param_name, param_value in parameters.items(): 
                if hasattr(strategy, param_name): 
                    setattr(strategy, param_name, param_value)
            
        except Exception as e: 
            self.logger.error(f"Error updating strategy parameters: {e}")
    
    def _calculate_score(self, backtest_results: BacktestResults, metric: OptimizationMetric)->float:
        """Calculate optimization score"""
        try: 
            if metric ==  OptimizationMetric.SHARPE_RATIO: 
                return backtest_results.sharpe_ratio
            elif metric  ==  OptimizationMetric.CALMAR_RATIO: 
                return backtest_results.calmar_ratio
            elif metric  ==  OptimizationMetric.TOTAL_RETURN: 
                return backtest_results.total_return
            elif metric  ==  OptimizationMetric.MAX_DRAWDOWN: 
                return -backtest_results.max_drawdown  # Negative because we want to minimize drawdown
            elif metric  ==  OptimizationMetric.WIN_RATE: 
                return backtest_results.win_rate
            elif metric  ==  OptimizationMetric.PROFIT_FACTOR: 
                return backtest_results.profit_factor
            else: 
                return backtest_results.sharpe_ratio  # Default to Sharpe ratio
            
        except Exception as e: 
            self.logger.error(f"Error calculating score: {e}")
            return 0.0


class OptimizationAnalyzer: 
    """Optimization results analyzer"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
    
    def analyze_optimization(self, result: OptimizationResult)->Dict[str, Any]: 
        """Analyze optimization results"""
        try: 
            analysis = {
                "optimization_summary": {
                    "method": "optimization_completed",
                    "iterations": result.iterations_completed,
                    "optimization_time": f"{result.optimization_time:.2f} seconds",
                    "best_score": f"{result.best_score:.4f}"
                },
                "best_parameters": result.best_parameters,
                "best_performance": {
                    "total_return": f"{result.best_results.total_return:.2%}",
                    "annualized_return": f"{result.best_results.annualized_return:.2%}",
                    "sharpe_ratio": f"{result.best_results.sharpe_ratio:.2f}",
                    "max_drawdown": f"{result.best_results.max_drawdown:.2%}",
                    "win_rate": f"{result.best_results.win_rate:.2%}",
                    "profit_factor": f"{result.best_results.profit_factor:.2f}"
                },
                "parameter_sensitivity": self._analyze_parameter_sensitivity(result.all_results),
                "convergence_analysis": self._analyze_convergence(result.all_results)
            }
            
            return analysis
            
        except Exception as e: 
            self.logger.error(f"Error analyzing optimization: {e}")
            return {"error": str(e)}
    
    def _analyze_parameter_sensitivity(self, all_results: List[Tuple[Dict[str, Any], float, BacktestResults]])->Dict[str, Any]: 
        """Analyze parameter sensitivity"""
        try: 
            sensitivity = {}
            
            # Get all parameter names
            if all_results: 
                param_names = list(all_results[0][0].keys())
                
                for param_name in param_names: 
                    param_values = [result[0][param_name] for result in all_results]
                    scores = [result[1] for result in all_results]
                    
                    # Calculate correlation between parameter and score
                    correlation = self._calculate_correlation(param_values, scores)
                    
                    sensitivity[param_name] = {
                        "correlation": correlation,
                        "min_value": min(param_values),
                        "max_value": max(param_values),
                        "best_value": param_values[scores.index(max(scores))]
                    }
            
            return sensitivity
            
        except Exception as e: 
            self.logger.error(f"Error analyzing parameter sensitivity: {e}")
            return {}
    
    def _analyze_convergence(self, all_results: List[Tuple[Dict[str, Any], float, BacktestResults]])->Dict[str, Any]: 
        """Analyze optimization convergence"""
        try: 
            scores = [result[1] for result in all_results]
            
            convergence = {
                "initial_score": scores[0] if scores else 0,
                "final_score": scores[-1] if scores else 0,
                "improvement": scores[-1] - scores[0] if scores else 0,
                "best_score": max(scores) if scores else 0,
                "convergence_rate": self._calculate_convergence_rate(scores)
            }
            
            return convergence
            
        except Exception as e: 
            self.logger.error(f"Error analyzing convergence: {e}")
            return {}
    
    def _calculate_correlation(self, x: List[float], y: List[float])->float:
        """Calculate correlation coefficient"""
        try: 
            if len(x)  !=  len(y) or len(x)  <  2: 
                return 0.0
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
            
            if denominator ==  0: 
                return 0.0
            
            return numerator / denominator
            
        except Exception as e: 
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_convergence_rate(self, scores: List[float])->float:
        """Calculate convergence rate"""
        try: 
            if len(scores)  <  2: 
                return 0.0
            
            # Calculate improvement over iterations
            improvements = []
            for i in range(1, len(scores)): 
                improvement = scores[i] - scores[i - 1]
                improvements.append(improvement)
            
            # Calculate average improvement rate
            if improvements: 
                return sum(improvements) / len(improvements)
            else: 
                return 0.0
            
        except Exception as e: 
            self.logger.error(f"Error calculating convergence rate: {e}")
            return 0.0
