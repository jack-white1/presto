// Jack White 2023, jack.white@eng.ox.ac.uk

// This file is used to profile the candidate_sigma function.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "accel.h"


void profile_candidate_sigma(double power_min, double power_max, double number_of_power_steps, 
                                int numsum_min, int numsum_max, int number_of_numsum_steps, 
                                double independent_trials_min, double independent_trials_max, double number_of_independent_trials_steps){
    FILE *profile_file = fopen("profile.csv", "w");
    double power_step = (power_max - power_min) / number_of_power_steps;
    int numsum_step = (numsum_max - numsum_min) / number_of_numsum_steps;
    double independent_trials_step = (independent_trials_max - independent_trials_min) / number_of_independent_trials_steps;
    for (double power = power_min; power < power_max; power = power + power_step){
        for (int numsum = numsum_min; numsum < numsum_max; numsum = numsum + numsum_step){
            for (double independent_trials = independent_trials_min; independent_trials < independent_trials_max; independent_trials = independent_trials + independent_trials_step){
                double sigma = candidate_sigma(power, numsum, independent_trials);
                fprintf(profile_file, "%lf,%lf,%d,%lf\n", sigma, power, numsum, independent_trials);
            }
        }
    }
}

//int main(int argc, char *argv[]) {
int test_profile(int argc, char *argv[]) {

    if (argc < 2) {
        printf("USAGE: %s [-power_min double] [-power_max double] [-number_power_steps double] \n", argv[0]);
        printf("          [-numsum_min int] [-numsum_max int] [-number_numsum_steps int] \n");
        printf("          [-independent_trials_min double] [-independent_trials_max double] [-number_independent_trials_steps double] \n");
        printf("Optional arguments:\n");
        printf("  -power_min: Minimum power to profile (default = 0.0)\n");
        printf("  -power_max: Maximum power to profile (default = 10000.0)\n");
        printf("  -number_power_steps: Number of power steps to profile (default = 1000)\n");
        printf("  -numsum_min: Minimum numsum to profile (default = 2)\n");
        printf("  -numsum_max: Maximum numsum to profile (default = 1200)\n");
        printf("  -number_numsum_steps: Number of numsum steps to profile (default = 120)\n");
        printf("  -independent_trials_min: Minimum independent_trials to profile (default = 1.0)\n");
        printf("  -independent_trials_max: Maximum independent_trials to profile (default = 100000.0)\n");
        printf("  -number_independent_trials_steps: Number of independent_trials steps to profile (default = 1000)\n");
        
        return 1;
    }

    // Get the power_min value from the command line arguments
    // If not provided, default to 0.0
    double power_min = 0.0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-power_min") == 0 && i+1 < argc) {
            power_min = atoi(argv[i+1]);
        }
    }

    // Get the power_max value from the command line arguments
    // If not provided, default to 10000.0
    double power_max = 10000.0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-power_max") == 0 && i+1 < argc) {
            power_max = atoi(argv[i+1]);
        }
    }

    // Get the number_of_power_steps value from the command line arguments
    // If not provided, default to 1000
    double number_of_power_steps = 1000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-number_power_steps") == 0 && i+1 < argc) {
            number_of_power_steps = atoi(argv[i+1]);
        }
    }

    // Get the numsum_min value from the command line arguments
    // If not provided, default to 2
    int numsum_min = 2;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-numsum_min") == 0 && i+1 < argc) {
            numsum_min = atoi(argv[i+1]);
        }
    }

    // Get the numsum_max value from the command line arguments
    // If not provided, default to 1200
    int numsum_max = 1200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-numsum_max") == 0 && i+1 < argc) {
            numsum_max = atoi(argv[i+1]);
        }
    }

    // Get the number_of_numsum_steps value from the command line arguments
    // If not provided, default to 120
    int number_of_numsum_steps = 120;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-number_numsum_steps") == 0 && i+1 < argc) {
            number_of_numsum_steps = atoi(argv[i+1]);
        }
    }

    // Get the independent_trials_min value from the command line arguments
    // If not provided, default to 1.0
    double independent_trials_min = 1.0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-independent_trials_min") == 0 && i+1 < argc) {
            independent_trials_min = atoi(argv[i+1]);
        }
    }

    // Get the independent_trials_max value from the command line arguments
    // If not provided, default to 100000.0
    double independent_trials_max = 100000.0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-independent_trials_max") == 0 && i+1 < argc) {
            independent_trials_max = atoi(argv[i+1]);
        }
    }

    // Get the number_of_independent_trials_steps value from the command line arguments
    // If not provided, default to 1000
    double number_of_independent_trials_steps = 1000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-number_independent_trials_steps") == 0 && i+1 < argc) {
            number_of_independent_trials_steps = atoi(argv[i+1]);
        }
    }

    profile_candidate_sigma(power_min, power_max, number_of_power_steps, 
                                numsum_min, numsum_max, number_of_numsum_steps, 
                                independent_trials_min, independent_trials_max, number_of_independent_trials_steps);
    return 0;
}