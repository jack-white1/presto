// Jack White 2023, jack.white@eng.ox.ac.uk

// This program reads in a .fft file produced by PRESTO realfft
// and computes the boxcar filter candidates for a range of boxcar widths, 1 to zmax (default 1200)
// The number of candidates per boxcar is set by the user, default 10

// The output is a text/binary file called INPUTFILENAME.bctxtcand (or .bccand for binary) with the following columns:
// sigma,power,period_ms,frequency,frequency_index,fdot,boxcar_width,acceleration

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "accel.h"
//#include "presto.h"

#define SPEED_OF_LIGHT 299792458.0

typedef struct {
    double sigma;
    float power;
    float period_ms;
    float frequency;
    int frequency_index;
    float fdot;
    int boxcar_width;
    float acceleration;
} Candidate;

int compare_candidates(const void *a, const void *b) {
    Candidate *candidateA = (Candidate *)a;
    Candidate *candidateB = (Candidate *)b;
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

float* compute_magnitude_block_normalization_mad(const char *filepath, int *magnitude_size) {
    size_t block_size = 32768; // needs to be much larger than max boxcar width

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
    return magnitude;
}

void recursive_boxcar_filter(float* magnitudes_array, int magnitudes_array_length, int max_boxcar_width, const char *filename, int candidates_per_boxcar, float observation_time_seconds, float sigma_threshold, int z_step) {
    //printf("Computing boxcar filter candidates for %d boxcar widths...\n", max_boxcar_width);

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s.txtcand", base_name);
    //printf("Storing up to %d candidates per boxcar in text format in %s\n", candidates_per_boxcar, text_filename);

    FILE *text_candidates_file = fopen(text_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return;
    }
    fprintf(text_candidates_file, "sigma,power,period[ms],frequency[hz],frequency_index[bin],fdot[hz/s],boxcar_width[bins],acceleration[m/s^2]\n");

    
    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array++;
    magnitudes_array_length--;

    int valid_length = magnitudes_array_length;
    int initial_length = magnitudes_array_length;
    int offset = 0;

    // prepare output array
    float* output_array = (float*) malloc(sizeof(float) * magnitudes_array_length);


    //Candidate candidates[max_boxcar_width][candidates_per_boxcar];
    Candidate* candidates = (Candidate*) malloc(sizeof(Candidate) * max_boxcar_width * candidates_per_boxcar);

    // set candidates array to all zeros
    #pragma omp parallel for
    for (int i = 0; i < max_boxcar_width * candidates_per_boxcar; i++){
        candidates[i].sigma = 0.0;
        candidates[i].power = 0.0;
        candidates[i].period_ms = 0.0;
        candidates[i].frequency = 0.0;
        candidates[i].frequency_index = 0;
        candidates[i].fdot = 0.0;
        candidates[i].boxcar_width = 0;
        candidates[i].acceleration = 0.0;
    }

    valid_length = magnitudes_array_length;
    offset = 0;

    // begin timer
    double start = omp_get_wtime();


    for (int boxcar_width = 2; boxcar_width < max_boxcar_width; boxcar_width++) {
        //printf("Computing boxcar width %d\n", boxcar_width);
        valid_length -= 1;
        offset += 1;

        #pragma omp parallel for
        for (int i = 0; i < valid_length; i++) {
            output_array[i] += magnitudes_array[i + offset];
        }

        if (boxcar_width % z_step == 0){
            int window_length = valid_length / candidates_per_boxcar;
            if (candidates_per_boxcar > 0){
                #pragma omp parallel for
                for (int i = 0; i < candidates_per_boxcar; i++) {
                    float local_max_power = -INFINITY;
                    int window_start = i * window_length;

                    // initialise the candidate
                    int candidate_index = (boxcar_width-2)*candidates_per_boxcar + i;
                    candidates[candidate_index].sigma = 0.0;
                    candidates[candidate_index].power = 0.0;
                    candidates[candidate_index].period_ms = 0.0;
                    candidates[candidate_index].frequency = 0.0;
                    candidates[candidate_index].frequency_index = 0;
                    candidates[candidate_index].fdot = 0.0;
                    candidates[candidate_index].boxcar_width = 0;
                    candidates[candidate_index].acceleration = 0.0;

                    for (int j = window_start; j < window_start + window_length; j++){
                        if (output_array[j] > local_max_power) {
                            local_max_power = output_array[j];
                            candidates[candidate_index].frequency_index = j;
                            candidates[candidate_index].power = local_max_power;
                            candidates[candidate_index].boxcar_width = boxcar_width;
                            if (observation_time_seconds > 0) {
                                candidates[candidate_index].frequency = frequency_from_observation_time_seconds(observation_time_seconds,candidates[candidate_index].frequency_index);
                                candidates[candidate_index].period_ms = period_ms_from_frequency(candidates[candidate_index].frequency);
                                candidates[candidate_index].fdot = fdot_from_boxcar_width(candidates[candidate_index].boxcar_width, observation_time_seconds);
                                candidates[candidate_index].acceleration = acceleration_from_fdot(candidates[candidate_index].fdot, candidates[candidate_index].frequency);
                            }
                        }
                    }
                    double num_independent_trials = ((double)max_boxcar_width)*((double)initial_length)/6.95; // 6.95 from eqn 6 in Anderson & Ransom 2018
                    candidates[candidate_index].sigma = candidate_sigma(candidates[candidate_index].power*0.5, candidates[candidate_index].boxcar_width, num_independent_trials); 
                }
            }
        }
    }

    // end timer
    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("\tCalculating and searching the filters took %f seconds\n", time_spent);

    // begin timer for writing output file
    start = omp_get_wtime();

    if (candidates_per_boxcar > 0){
        qsort(candidates, candidates_per_boxcar*max_boxcar_width, sizeof(Candidate), compare_candidates);

        for (int i = 0; i < max_boxcar_width*candidates_per_boxcar; i++){
            if (candidates[i].sigma > sigma_threshold ){
                fprintf(text_candidates_file, "%lf,%f,%f,%f,%d,%f,%d,%f\n", 
                    candidates[i].sigma,
                    candidates[i].power,
                    candidates[i].period_ms,
                    candidates[i].frequency,
                    candidates[i].frequency_index,
                    candidates[i].fdot,
                    candidates[i].boxcar_width,
                    candidates[i].acceleration);
            }
        }
    }

    end = omp_get_wtime();
    time_spent = end - start;
    printf("\tWriting output file took %f seconds\n", time_spent);

    fclose(text_candidates_file);
    free(base_name);
    free(candidates);
    free(output_array);
}

void recursive_boxcar_filter_cache_optimised(float* magnitudes_array, int magnitudes_array_length, int max_boxcar_width, const char *filename, int candidates_per_boxcar, float observation_time_seconds, float sigma_threshold, int z_step, int block_width) {

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s.txtcand", base_name);

    FILE *text_candidates_file = fopen(text_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return;
    }
    fprintf(text_candidates_file, "sigma,power,rbin,zbin\n");

    
    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array++;
    magnitudes_array_length--;

    int valid_length = magnitudes_array_length;
    int initial_length = magnitudes_array_length;

    valid_length = magnitudes_array_length;
    double num_independent_trials = ((double)max_boxcar_width)*((double)initial_length)/6.95; // 6.95 from eqn 6 in Anderson & Ransom 2018

    int zmax = max_boxcar_width;

    int num_blocks = (valid_length + block_width - 1) / block_width;

    float* max_array = (float*) malloc(sizeof(float) *  num_blocks * zmax);
    //memset max array to zero
    memset(max_array, 0.0, sizeof(float) * num_blocks * zmax);

    long* max_index_array = (long*) malloc(sizeof(long) *  num_blocks * zmax);
    //memset max index array to zero
    memset(max_index_array, 0, sizeof(long) * num_blocks * zmax);


    // begin timer for boxcar filtering
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int block_index = 0; block_index < num_blocks; block_index++){
        float* lookup_array = (float*) malloc(sizeof(float) *  (block_width + zmax));
        float* sum_array = (float*) malloc(sizeof(float) *  block_width);

        // memset lookup array and sum array to zero
        memset(lookup_array, 0.0, sizeof(float) * (block_width + zmax));

        // initialise lookup array
        for (int i = 0; i < block_width + zmax; i++){
            if (i + block_index*block_width < valid_length){
                lookup_array[i] = magnitudes_array[i + block_index*block_width];
            }
        }

        // initialise sum array
        for (int i = 0; i < block_width; i++){
            sum_array[i] = lookup_array[i];
        }

        float local_max_power = -INFINITY;
        long local_max_index = 0;

        // periodicity search the sum array
        for (int i = 0; i < block_width; i++){
            if (sum_array[i] > local_max_power) {
                local_max_power = sum_array[i];
                local_max_index = i;
            }
        }
        max_array[block_index*zmax] = local_max_power;
        max_index_array[block_index*zmax] = local_max_index + block_index*block_width;

        for (int z = 1; z < zmax; z++){
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
                        local_max_index = i;
                    }
                }
                max_array[block_index*zmax + z] = local_max_power;
                max_index_array[block_index*zmax + z] = local_max_index + block_index*block_width;
            }
        }
    }

    // end timer for boxcar filtering
    double end = omp_get_wtime();

    double time_spent = end - start;
    printf("\tCalculating and searching the filters took %f seconds\n", time_spent);


    // begin timer for writing output file
    start = omp_get_wtime();
    // write out max_array to text file
    for (int i = 0; i < num_blocks; i++){
        for (int z = 0; z < zmax; z++){
            if (z % z_step == 0){
                fprintf(text_candidates_file, "%f,%f,%ld,%d\n", 
                    candidate_sigma(max_array[i*zmax + z]*0.5, z, num_independent_trials),
                    max_array[i*zmax + z],
                    max_index_array[i*zmax + z],
                    z);
            }
        }
    }
    fclose(text_candidates_file);
    free(base_name);
    free(max_array);
    end = omp_get_wtime();
    time_spent = end - start;
    printf("\tWriting output file took %f seconds\n", time_spent);
}



int main(int argc, char *argv[]) {
    // start overall program timer
    double start_program = omp_get_wtime();

    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-candidates int] [-tobs float] [-sigma float] [-zstep int] [-cache_optimised] [-block_width int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-ncpus [int]\tThe number of OpenMP threads to use (default 1)\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 1200, max = the size of your input data)\n");
        printf("\t-candidates [int]\tThe number of candidates per boxcar (default = 10), total candidates in output will be = zmax * candidates\n");
        printf("\t-tobs [float]\tThe observation time (default = 0.0), this must be specified if you want accurate frequency/acceleration values\n");
        printf("\t-sigma [float]\tThe sigma threshold (default = 0.0), candidates with sigma below this value will not be written to the output files\n");
        printf("\t-zstep [int]\tThe step size in z (default = 2).\n");
        printf("\t-cache_optimised\tUse the faster cache optimised version of the algorithm (default = false)\n");
        printf("\t-block_width\t The block width to use for the cache optimised version of the algorithm (default = 10000)\n");
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
    int num_threads = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-ncpus") == 0 && i+1 < argc) {
            num_threads = atoi(argv[i+1]);
        }
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 1200
    int max_boxcar_width = 1200;
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
    // If not provided, default to 0.0
    float sigma_threshold = 0.0f;
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

    // Get the cache optimised flag from the command line arguments
    // If not provided, default to false
    int cache_optimised = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-cache_optimised") == 0) {
            cache_optimised = 1;
        }
    }

    // Get the block width from the command line arguments
    // If not provided, default to 10000
    int block_width = 10000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-block_width") == 0 && i+1 < argc) {
            block_width = atoi(argv[i+1]);
        }
    }

    omp_set_num_threads(num_threads);

    // begin timer for reading input file
    double start = omp_get_wtime();
    int magnitude_array_size;
    float* magnitudes = compute_magnitude_block_normalization_mad(argv[1], &magnitude_array_size);

    if(magnitudes == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }
    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Reading and normalising input file took %f seconds\n", time_spent);


    start = omp_get_wtime();
    if (cache_optimised == 0){
        recursive_boxcar_filter(magnitudes, 
            magnitude_array_size, 
            max_boxcar_width, 
            argv[1], 
            candidates_per_boxcar, 
            observation_time_seconds, 
            sigma_threshold,
            z_step);
    } else {
        recursive_boxcar_filter_cache_optimised(magnitudes, 
            magnitude_array_size, 
            max_boxcar_width, 
            argv[1], 
            candidates_per_boxcar, 
            observation_time_seconds, 
            sigma_threshold,
            z_step,
            block_width);
    }
    end = omp_get_wtime();
    time_spent = end - start;
    printf("Time spent processing was %f seconds\n", time_spent);


    free(magnitudes);

    // end overall program timer
    double end_program = omp_get_wtime();
    double time_spent_program = end_program - start_program;
    printf("Total time spent was %f seconds\n", time_spent_program);
    return 0;
}