// Jack White 2023, jack.white@eng.ox.ac.uk

// This program reads in a .fft file produced by PRESTO realfft
// and computes the boxcar filter candidates for a range of boxcar widths, 0 to zmax (default 200)
// The number of candidates per boxcar is set by the user, default 10


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "accel.h"


#define SPEED_OF_LIGHT 299792458.0

// ANSI Color Codes
#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"


typedef struct {
    double sigma;
    float power;
    long index;
    int z;
} cache_optimised_candidate;

int compare_cache_optimised_candidates_power(const void *a, const void *b) {
    cache_optimised_candidate *candidateA = (cache_optimised_candidate *)a;
    cache_optimised_candidate *candidateB = (cache_optimised_candidate *)b;
    if(candidateA->power > candidateB->power) return -1; // for descending order
    if(candidateA->power < candidateB->power) return 1;
    return 0;
}

int compare_cache_optimised_candidates_sigma(const void *a, const void *b) {
    cache_optimised_candidate *candidateA = (cache_optimised_candidate *)a;
    cache_optimised_candidate *candidateB = (cache_optimised_candidate *)b;
    if(candidateA->sigma > candidateB->sigma) return -1; // for descending order
    if(candidateA->sigma < candidateB->sigma) return 1;
    return 0;
}

float fdot_from_boxcar_width(int boxcar_width, float observation_time_seconds){
    return boxcar_width / (observation_time_seconds*observation_time_seconds);
}

float acceleration_from_fdot(float fdot, float frequency){
    return fdot * SPEED_OF_LIGHT / frequency;
}

float frequency_from_observation_time_seconds(float observation_time_seconds, int frequency_index){
    return frequency_index / observation_time_seconds;
}

float period_ms_from_frequency(float frequency){
    return 1000.0 / frequency;
}

// function to compare floats for qsort
int compare_floats_median(const void *a, const void *b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;

    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

void normalize_block(float* block, size_t block_size) {
    if (block_size == 0) return;

    // Compute the median
    float* sorted_block = (float*) malloc(sizeof(float) * block_size);
    memcpy(sorted_block, block, sizeof(float) * block_size);
    qsort(sorted_block, block_size, sizeof(float), compare_floats_median);

    float median;
    if (block_size % 2 == 0) {
        median = (sorted_block[block_size/2 - 1] + sorted_block[block_size/2]) / 2.0f;
    } else {
        median = sorted_block[block_size/2];
    }

    // Compute the MAD
    for (size_t i = 0; i < block_size; i++) {
        sorted_block[i] = fabs(sorted_block[i] - median);
    }
    qsort(sorted_block, block_size, sizeof(float), compare_floats_median);

    float mad = block_size % 2 == 0 ?
                (sorted_block[block_size/2 - 1] + sorted_block[block_size/2]) / 2.0f :
                sorted_block[block_size/2];

    free(sorted_block);

    // scale the mad by the constant scale factor k
    float k = 1.4826f; // 1.4826 is the scale factor to convert mad to std dev for a normal distribution https://en.wikipedia.org/wiki/Median_absolute_deviation
    mad *= k;

    // Normalize the block
    if (mad != 0) {
        for (size_t i = 0; i < block_size; i++) {
            block[i] = (block[i] - median) / mad;
        }
    }

}

float* compute_magnitude_block_normalization_mad(const char *filepath, int *magnitude_size, int ncpus, int max_boxcar_width) {
    // begin timer for reading input file
    double start = omp_get_wtime();
    size_t block_size = max_boxcar_width * 30; // needs to be much larger than max boxcar width

    //printf("Reading file: %s\n", filepath);

    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Determine the size of the file
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t num_floats = filesize / sizeof(float);

    // Allocate memory for the data
    float* data = (float*) malloc(sizeof(float) * num_floats);
    if(data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    size_t n = fread(data, sizeof(float), num_floats, f);
    if (n % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        free(data);
        return NULL;
    }

    size_t size = n / 2;
    float* magnitude = (float*) malloc(sizeof(float) * size);
    if(magnitude == NULL) {
        printf("Memory allocation failed\n");
        free(data);
        return NULL;
    }

    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Reading the data took      %f seconds using 1 thread\n", time_spent);

    start = omp_get_wtime();

    #pragma omp parallel for
    // Perform block normalization
    for (size_t block_start = 0; block_start < size; block_start += block_size) {
        size_t block_end = block_start + block_size < size ? block_start + block_size : size;
        size_t current_block_size = block_end - block_start;

        // Separate the real and imaginary parts
        float* real_block = (float*) malloc(sizeof(float) * current_block_size);
        float* imag_block = (float*) malloc(sizeof(float) * current_block_size);

        if (real_block == NULL || imag_block == NULL) {
            printf("Memory allocation failed for real_block or imag_block\n");
            free(real_block);
            free(imag_block);
        }

        for (size_t i = 0; i < current_block_size; i++) {
            real_block[i] = data[2 * (block_start + i)];
            imag_block[i] = data[2 * (block_start + i) + 1];
        }

        // Normalize real and imaginary parts independently
        normalize_block(real_block, current_block_size);
        normalize_block(imag_block, current_block_size);

        // Recompute the magnitudes after normalization
        for (size_t i = block_start; i < block_end; i++) {
            magnitude[i] = real_block[i - block_start] * real_block[i - block_start] +
                        imag_block[i - block_start] * imag_block[i - block_start];
        }

        free(real_block);
        free(imag_block);
    }

    magnitude[0] = 0.0f; // set DC component of magnitude spectrum to 0

    fclose(f);
    free(data);

    *magnitude_size = (int) size;

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Normalizing the data took  %f seconds using %d thread(s)\n", time_spent, ncpus);
    return magnitude;
}

void recursive_boxcar_filter_cache_optimised(float* magnitudes_array, int magnitudes_array_length, \
                                int max_boxcar_width, const char *filename, int candidates_per_boxcar, \
                                float observation_time_seconds, float sigma_threshold, int z_step, \
                                int block_width, int ncpus) {

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s.bctxtcand", base_name);

    FILE *text_candidates_file = fopen(text_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return;
    }
    fprintf(text_candidates_file, "sigma, power, period, frequency, rbin, f-dot, z, acceleration\n");

    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array++;
    magnitudes_array_length--;

    int valid_length = magnitudes_array_length;
    int initial_length = magnitudes_array_length;

    valid_length = magnitudes_array_length;
    double num_independent_trials = ((double)max_boxcar_width)*((double)initial_length)/6.95; // 6.95 from eqn 6 in Anderson & Ransom 2018

    int zmax = max_boxcar_width;

    int num_blocks = (valid_length + block_width - 1) / block_width;

    cache_optimised_candidate* candidates = (cache_optimised_candidate*) malloc(sizeof(cache_optimised_candidate) *  num_blocks * zmax);
    //memset cache_optimised_candidates to zero
    memset(candidates, 0, sizeof(cache_optimised_candidate) * num_blocks * zmax);
    
    // begin timer for boxcar filtering
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int block_index = 0; block_index < num_blocks; block_index++) {
        float* lookup_array = (float*) malloc(sizeof(float) * (block_width + zmax));
        float* sum_array = (float*) malloc(sizeof(float) * block_width);

        // memset lookup array and sum array to zero
        memset(lookup_array, 0, sizeof(float) * (block_width + zmax));
        memset(sum_array, 0, sizeof(float) * block_width);

        // initialise lookup array
        int num_to_copy = block_width + zmax;
        if (block_index * block_width + num_to_copy > valid_length) {
            num_to_copy = valid_length - block_index * block_width;
        }
        memcpy(lookup_array, magnitudes_array + block_index * block_width, sizeof(float) * num_to_copy);

        // memset sum array to 0
        memset(sum_array, 0, sizeof(float) * block_width);


        float local_max_power;
        long local_max_index;
        
        for (int z = 0; z < zmax; z++){
            // boxcar filter
            for (int i = 0; i < block_width; i++){
                sum_array[i] += lookup_array[i + z];
            }

            // find max
            if (z % z_step == 0){
                local_max_power = -INFINITY;
                local_max_index = 0;
                for (int i = 0; i < block_width; i++){
                    if (sum_array[i] > local_max_power) {
                        local_max_power = sum_array[i];
                        local_max_index = (long)i + (long)block_index*(long)block_width;
                    }
                }
                candidates[num_blocks*z + block_index].power = local_max_power;
                candidates[num_blocks*z + block_index].index = local_max_index;
            }
        }
    }

    // end timer for boxcar filtering
    double end = omp_get_wtime();

    double time_spent = end - start;
    printf("Searching the data took    %f seconds using %d thread(s)\n", time_spent, ncpus);

    start = omp_get_wtime();

    cache_optimised_candidate* final_output_candidates = (cache_optimised_candidate*) malloc(sizeof(cache_optimised_candidate) *  candidates_per_boxcar * zmax);

    // memset final_output_candidates to zero
    memset(final_output_candidates, 0, sizeof(cache_optimised_candidate) * candidates_per_boxcar * zmax);

    float temp_sigma;
    // extract candidates_per_boxcar candidates from max_array
    //#pragma omp parallel
    for (int z = 0; z < zmax; z+=z_step){
        cache_optimised_candidate* local_candidates = (cache_optimised_candidate*) malloc(sizeof(cache_optimised_candidate) *  num_blocks);
        // extract the row from candidates using memcpy
        memcpy(local_candidates, candidates + z*num_blocks, sizeof(cache_optimised_candidate) * num_blocks);

        // sort the row by descending .power values using qsort
        qsort(local_candidates, num_blocks, sizeof(cache_optimised_candidate), compare_cache_optimised_candidates_power);

        // write the top candidates_per_boxcar candidates to final_output_candidates, if they are above the sigma threshold
        for (int i = 0; i < candidates_per_boxcar; i++){
            temp_sigma = candidate_sigma(local_candidates[i].power*0.5, z, num_independent_trials);
            if (temp_sigma > sigma_threshold){
                if (local_candidates[i].index > 0){
                    final_output_candidates[z*candidates_per_boxcar + i].sigma = temp_sigma;
                    final_output_candidates[z*candidates_per_boxcar + i].power = local_candidates[i].power;
                    final_output_candidates[z*candidates_per_boxcar + i].index = local_candidates[i].index;
                    final_output_candidates[z*candidates_per_boxcar + i].z = z;
                }      
            }
        }
    }

    // sort final_output_candidates by descending sigma using qsort
    qsort(final_output_candidates, candidates_per_boxcar * zmax, sizeof(cache_optimised_candidate), compare_cache_optimised_candidates_sigma);

    // dump final_output_candidates to binary file
    char binary_filename[255];
    snprintf(binary_filename, 255, "%s.bccand", base_name);
    FILE *binary_candidates_file = fopen(binary_filename, "wb"); // open the file for writing. Make sure you have write access in this directory.
    if (binary_candidates_file == NULL) {
        printf("Could not open file for writing binary results.\n");
        return;
    }
    fwrite(final_output_candidates, sizeof(cache_optimised_candidate), candidates_per_boxcar * zmax, binary_candidates_file);
    fclose(binary_candidates_file);

    float temp_period_ms;
    float temp_frequency;
    float temp_fdot;
    float temp_acceleration;
    // write final_output_candidates to text file with physical measurements
    for (int i = 0; i < candidates_per_boxcar * zmax; i++){
        if (final_output_candidates[i].sigma > sigma_threshold){
            temp_period_ms = period_ms_from_frequency(frequency_from_observation_time_seconds(observation_time_seconds,final_output_candidates[i].index));
            temp_frequency = frequency_from_observation_time_seconds(observation_time_seconds,final_output_candidates[i].index);
            temp_fdot = fdot_from_boxcar_width(final_output_candidates[i].z, observation_time_seconds);
            temp_acceleration = acceleration_from_fdot(fdot_from_boxcar_width(final_output_candidates[i].z, observation_time_seconds), frequency_from_observation_time_seconds(observation_time_seconds,final_output_candidates[i].index));
            fprintf(text_candidates_file, "%lf,%f,%f,%f,%ld,%f,%d,%f\n", 
                final_output_candidates[i].sigma,
                final_output_candidates[i].power,
                temp_period_ms,
                temp_frequency,
                final_output_candidates[i].index,
                temp_fdot,
                final_output_candidates[i].z,
                temp_acceleration);
        }
    }

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Producing output took      %f seconds using 1 thread\n", time_spent);

    fclose(text_candidates_file);
    free(base_name);
    free(candidates);
    free(final_output_candidates);

}

void profile_candidate_sigma(){
    // open csv file for writing
    FILE *csv_file = fopen("candidate_sigma_profile.csv", "w"); // open the file for writing. Make sure you have write access in this directory.
    if (csv_file == NULL) {
        printf("Could not open file for writing candidate sigma profile.\n");
        return;
    }
    fprintf(csv_file, "sigma, power, z, num_independent_trials\n");

    for (int num_independent_trials = 65536; num_independent_trials < 1073741824; num_independent_trials*=2){
        for (int z = 1; z < 1200; z++){
            printf("z = %d\n", z);
            for (double target_sigma = 1.0; target_sigma < 30.0; target_sigma+=1.0){
                printf("target_sigma = %lf\n", target_sigma);
                // increase power in steps of 0.1 until output sigma is above target sigma
                double power = 0.0;
                double output_sigma = 0.0;
                while (output_sigma < target_sigma){
                    power += 0.1;
                    output_sigma = candidate_sigma(power*0.5, z, num_independent_trials);
                }
                fprintf(csv_file, "%lf,%lf,%d,%d\n", output_sigma, power, z, num_independent_trials);
                printf("z = %d, power = %lf, output_sigma = %lf, num_independent = %d\n", z, power, output_sigma, num_independent_trials);
            }
        }
    }
    fclose(csv_file);
}


const char* pulscan_frame = 
"    .          .     .     *        .   .   .     .\n"
"         "BOLD"___________      . __"RESET" .  .   *  .   .  .  .     .\n"            
"    . *   "BOLD"_____  __ \\__+ __/ /_____________ _____"RESET" .    "FLASHING"*"RESET"  .\n"
"  +    .   "BOLD"___  /_/ / / / / / ___/ ___/ __ `/ __ \\"RESET"     + .\n"
" .          "BOLD"_  ____/ /_/ / (__  ) /__/ /_/ / / / /"RESET" .  *     . \n"
"       .    "BOLD"/_/ *  \\__,_/_/____/\\___/\\__,_/_/ /_/"RESET"    \n"                         
"    *    +     .     .     . +     .     +   .      *   +\n"

"  J. White, K. Adámek, J. Roy, S. Ransom, W. Armour  2023\n\n";


int main(int argc, char *argv[]) {
    // start overall program timer
    double start_program = omp_get_wtime();
    printf("%s\n", pulscan_frame);

    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-candidates int] [-tobs float] [-sigma float] [-zstep int] [-block_width int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\t\tThe input file path (.fft file format like the output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-ncpus [int]\t\tThe number of OpenMP threads to use (default 1)\n");
        printf("\t-zmax [int]\t\tThe max boxcar width (default = 200, max = the size of your input data)\n");
        printf("\t-candidates [int]\tThe max number of candidates per boxcar (default = 10), total candidates in output will be less than or equal to [-zmax] * [-candidates]\n");
        printf("\t-tobs [float]\t\tThe observation time (default = 0.0), this must be specified if you want accurate frequency/acceleration values\n");
        printf("\t-sigma [float]\t\tThe sigma threshold (default = 1.0), candidates with sigma below this value will not be written to the output files\n");
        printf("\t-zstep [int]\t\tThe step size in z (default = 2).\n");
        printf("\t-block_width\t\tThe block width to use for the cache optimised version of the search algorithm (default = 32768)\n");
        printf("\t-candidate_sigma_profile\t\tProfile the candidate sigma function and write the results to candidate_sigma_profile.csv (you probably don't want to do this, default = 0)\n");
        return 1;
    }

    // Get the number of candidates per boxcar from the command line arguments
    // If not provided, default to 10
    int candidates_per_boxcar = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-candidates") == 0 && i+1 < argc) {
            candidates_per_boxcar = atoi(argv[i+1]);
        }
    }

    // Get the number of OpenMP threads from the command line arguments
    // If not provided, default to 1
    int ncpus = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-ncpus") == 0 && i+1 < argc) {
            ncpus = atoi(argv[i+1]);
        }
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 200
    int max_boxcar_width = 200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            max_boxcar_width = atoi(argv[i+1]);
        }
    }

    // Get the observation time from the command line arguments
    // If not provided, default to 0.0
    float observation_time_seconds = 0.0f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-tobs") == 0 && i+1 < argc) {
            observation_time_seconds = atof(argv[i+1]);
        }
    }

    if (observation_time_seconds == 0.0f) {
        printf("WARNING: No observation time provided, frequency and acceleration values will be inaccurate.\n");
        printf("[Optional] Please specify an observation time with the -tobs flag, e.g. -tobs 600.0\n");
    }

    // Get the sigma threshold value from the command line arguments
    // If not provided, default to 1.0
    float sigma_threshold = 1.0f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-sigma") == 0 && i+1 < argc) {
            sigma_threshold = atof(argv[i+1]);
        }
    }

    // Get the z step size from the command line arguments
    // If not provided, default to 2
    int z_step = 2;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zstep") == 0 && i+1 < argc) {
            z_step = atoi(argv[i+1]);
        }
    }

    // Get the candidate sigma profile flag from the command line arguments
    // If not provided, default to 0
    int candidate_sigma_profile = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-candidate_sigma_profile") == 0 && i+1 < argc) {
            candidate_sigma_profile = atoi(argv[i+1]);
        }
    }

    // Get the block width from the command line arguments
    // If not provided, default to 32768
    int block_width = 32768;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-block_width") == 0 && i+1 < argc) {
            block_width = atoi(argv[i+1]);
        }
    }

    if (candidate_sigma_profile > 0){
        profile_candidate_sigma();
        printf("Candidate sigma profile written to candidate_sigma_profile.csv\n");
        return 0;
    }

    omp_set_num_threads(ncpus);


    int magnitude_array_size;
    float* magnitudes = compute_magnitude_block_normalization_mad(argv[1], &magnitude_array_size, ncpus, max_boxcar_width);

    if(magnitudes == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }


    recursive_boxcar_filter_cache_optimised(magnitudes, 
            magnitude_array_size, 
            max_boxcar_width, 
            argv[1], 
            candidates_per_boxcar, 
            observation_time_seconds, 
            sigma_threshold,
            z_step,
            block_width,
            ncpus);

    free(magnitudes);

    // end overall program timer
    double end_program = omp_get_wtime();
    double time_spent_program = end_program - start_program;
    printf("--------------------------------------------\nTotal time spent was       " GREEN "%f seconds" RESET "\n\n\n", time_spent_program);

    //data written to file
    printf("Data written to .bctxtcand file (text format) and .bccand file (binary format)\n");


    return 0;
}