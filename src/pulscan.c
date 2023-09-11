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

#define MAX_DATA_SIZE 10000000000 // assume file won't be larger than this, 10M samples, increase if required
#define DEFAULT_CANDIDATES_PER_BOXCAR 10
#define SPEED_OF_LIGHT 299792458.0
#define DIFFERENCE_MULTIPLIER 9

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

float* compute_magnitude(const char *filepath, int *magnitude_size) {
    printf("Reading file: %s\n", filepath);

    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    float* data = (float*) malloc(sizeof(float) * MAX_DATA_SIZE);
    if(data == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }
    size_t n = fread(data, sizeof(float), MAX_DATA_SIZE, f);
    if (n % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        free(data);
        return NULL;
    }

    size_t size = n / 2;
    float* real_data = (float*) malloc(sizeof(float) * size);
    float* imag_data = (float*) malloc(sizeof(float) * size);

    for(size_t i = 0; i < size; i++) {
        real_data[i] = data[2 * i];
        imag_data[i] = data[2 * i + 1];
    }

    // compute mean and variance of real and imaginary components, ignoring DC component

    float real_sum = 0.0, imag_sum = 0.0;
    for(int i = 1; i < (int)n / 2; i++) {
        real_sum += data[2 * i];
        imag_sum += data[2 * i + 1];
    }
    float real_mean = real_sum / ((n-1) / 2);
    float imag_mean = imag_sum / ((n-1) / 2);

    float real_variance = 0.0, imag_variance = 0.0;
    for(int i = 1; i < (int)n / 2; i++) {
        real_variance += (data[2 * i] - real_mean)*(data[2 * i] - real_mean);
        imag_variance += (data[2 * i + 1] - imag_mean)*(data[2 * i + 1] - imag_mean);
    }
    real_variance /= ((n-1) / 2);
    imag_variance /= ((n-1) / 2);

    float real_stdev = sqrt(real_variance);
    float imag_stdev = sqrt(imag_variance);

    float* magnitude = (float*) malloc(sizeof(float) * n / 2);
    if(magnitude == NULL) {
        printf("Memory allocation failed\n");
        free(data);
        return NULL;
    }

    // set DC component of magnitude spectrum to 0
    magnitude[0] = 0.0f;

    float norm_real, norm_imag;

    for (int i = 1; i < (int) n / 2; i++) {
        norm_real = (data[2 * i] - real_mean) / real_stdev;
        norm_imag = (data[2 * i + 1] - imag_mean) / imag_stdev;
        magnitude[i] = norm_real*norm_real + norm_imag*norm_imag;
    }

    fclose(f);
    free(data);

    // pass the size of the magnitude array back through the output parameter
    *magnitude_size = (int) n / 2;

    // return the pointer to the magnitude array
    return magnitude;
}

void recursive_boxcar_filter(float* magnitudes_array, int magnitudes_array_length, int max_boxcar_width, const char *filename, int candidates_per_boxcar, float observation_time_seconds, float sigma_threshold, int output_boxcar_width) {
    printf("Computing boxcar filter candidates for %d boxcar widths...\n", max_boxcar_width);

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s.bctxtcand", base_name);
    printf("Storing %d candidates per boxcar in text format in %s\n", candidates_per_boxcar, text_filename);

    FILE *text_candidates_file = fopen(text_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return;
    }
    fprintf(text_candidates_file, "sigma,power,period[ms],frequency[hz],frequency_index[bin],fdot[hz/s],boxcar_width[bins],acceleration[m/s^2]\n");

    // Create new filename
    char binary_filename[255];
    snprintf(binary_filename, 255, "%s.bccand", base_name);
    printf("Storing %d candidates per boxcar in binary format in %s\n", candidates_per_boxcar, binary_filename);

    FILE *binary_candidates_file = fopen(binary_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (binary_candidates_file == NULL) {
        printf("Could not open file for writing binary results.\n");
        return;
    }
    
    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array++;
    magnitudes_array_length--;

    int valid_length = magnitudes_array_length;
    int initial_length = magnitudes_array_length;
    int offset = 0;

    // prepare output array
    float* output_array = (float*) malloc(sizeof(float) * magnitudes_array_length);


    //Candidate top_candidates[max_boxcar_width][candidates_per_boxcar];
    Candidate* top_candidates = (Candidate*) malloc(sizeof(Candidate) * max_boxcar_width * candidates_per_boxcar);

    valid_length = magnitudes_array_length;
    offset = 0;

    for (int boxcar_width = 2; boxcar_width < max_boxcar_width; boxcar_width++) {

        valid_length -= 1;
        offset += 1;

        #pragma omp parallel for
        for (int i = 0; i < valid_length; i++) {
            output_array[i] += magnitudes_array[i + offset];
        }

        int window_length = valid_length / candidates_per_boxcar;
        if (candidates_per_boxcar > 0){
            #pragma omp parallel for
            for (int i = 0; i < candidates_per_boxcar; i++) {
                float local_max_power = -INFINITY;
                int window_start = i * window_length;

                // initialise the candidate
                int candidate_index = boxcar_width*candidates_per_boxcar + i;
                top_candidates[candidate_index].sigma = 0.0;
                top_candidates[candidate_index].power = 0.0;
                top_candidates[candidate_index].period_ms = 0.0;
                top_candidates[candidate_index].frequency = 0.0;
                top_candidates[candidate_index].frequency_index = 0;
                top_candidates[candidate_index].fdot = 0.0;
                top_candidates[candidate_index].boxcar_width = 0;
                top_candidates[candidate_index].acceleration = 0.0;

                for (int j = window_start; j < window_start + window_length; j++){
                    if (output_array[j] > local_max_power) {
                        local_max_power = output_array[j];
                        top_candidates[candidate_index].frequency_index = j;
                        top_candidates[candidate_index].power = local_max_power;
                        top_candidates[candidate_index].boxcar_width = boxcar_width;
                        if (observation_time_seconds > 0) {
                            top_candidates[candidate_index].frequency = frequency_from_observation_time_seconds(observation_time_seconds,top_candidates[candidate_index].frequency_index);
                            top_candidates[candidate_index].period_ms = period_ms_from_frequency(top_candidates[candidate_index].frequency);
                            top_candidates[candidate_index].fdot = fdot_from_boxcar_width(top_candidates[candidate_index].boxcar_width, observation_time_seconds);
                            top_candidates[candidate_index].acceleration = acceleration_from_fdot(top_candidates[candidate_index].fdot, top_candidates[candidate_index].frequency);
                        }
                    }
                }
                top_candidates[candidate_index].sigma = candidate_sigma(top_candidates[candidate_index].power*0.5, top_candidates[candidate_index].boxcar_width, max_boxcar_width*initial_length);
            }
        }

        if (boxcar_width == output_boxcar_width){
            FILE *boxcar_file = fopen("boxcar_filtered_timeseries.dat", "wb");
            printf("Storing boxcar filtered timeseries in boxcar_filtered_timeseries.dat\n");
            if (boxcar_file == NULL) {
                printf("Could not open file for writing boxcar filtered timeseries.\n");
                return;
            }
            for (int i = 0; i < valid_length; i++) {
                fwrite(&output_array[i], sizeof(float), 1, boxcar_file);
            }
            fclose(boxcar_file);
        }
    }

    if (candidates_per_boxcar > 0){
        Candidate *all_candidates = (Candidate*) malloc(sizeof(Candidate) * max_boxcar_width * candidates_per_boxcar);
        for (int i = 2; i < max_boxcar_width; i++) {
            for (int j = 0; j < candidates_per_boxcar; j++) {
                all_candidates[i*candidates_per_boxcar + j] = top_candidates[i*candidates_per_boxcar + j];
            }
        }

        qsort(all_candidates, candidates_per_boxcar*max_boxcar_width, sizeof(Candidate), compare_candidates);

        for (int i = 2; i < max_boxcar_width*candidates_per_boxcar; i++){
            if (all_candidates[i].sigma > sigma_threshold ){
            fprintf(text_candidates_file, "%lf,%f,%f,%f,%d,%f,%d,%f\n", 
                all_candidates[i].sigma,
                all_candidates[i].power,
                all_candidates[i].period_ms,
                all_candidates[i].frequency,
                all_candidates[i].frequency_index,
                all_candidates[i].fdot,
                all_candidates[i].boxcar_width,
                all_candidates[i].acceleration);
            fwrite(&all_candidates[i], sizeof(Candidate), 1, binary_candidates_file);
            }
        }
    }
    fclose(text_candidates_file);
    fclose(binary_candidates_file);
    free(base_name);
    free(top_candidates);
    free(output_array);
}

void profile_candidate_sigma(){
    FILE *profile_file = fopen("profile.csv", "w");
    printf("sigma,power,boxcar_width,independent_trials\n");
    double independent_trials = 2500000000.0;
    for (double independent_trials = 1; independent_trials < 10000000; independent_trials = independent_trials + 10000){
        for (double power = 1; power < 10000; power = power + 200){
            for (int boxcar_width = 1; boxcar_width < 15000; boxcar_width = boxcar_width + 10){
                double sigma = candidate_sigma(power, boxcar_width, independent_trials);
                fprintf(profile_file, "%lf,%lf,%d,%lf\n", sigma, power, boxcar_width, independent_trials);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-candidates int] [-tobs float]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-ncpus [int]\tThe number of OpenMP threads to use (default 1)\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 1200, max = the size of your input data)\n");
        printf("\t-candidates [int]\tThe number of candidates per boxcar (default = 10), total candidates in output will be = zmax * candidates\n");
        printf("\t-tobs [float]\tThe observation time (default = 0.0), this must be specified if you want accurate frequency/acceleration values\n");
        printf("\t-sigma [float]\tThe sigma threshold (default = 0.0), candidates with sigma below this value will not be written to the output files\n");
        printf("\t-output_boxcar_width [int]\tThe boxcar width to output the boxcar filtered timeseries for (default = 0), if 0, no output will be written\n");
        return 1;
    }

    // Get the number of candidates per boxcar from the command line arguments
    // If not provided, default to 10
    int candidates_per_boxcar = DEFAULT_CANDIDATES_PER_BOXCAR;
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

    // Get the boxcar width to output the boxcar filtered timeseries for from the command line arguments
    // If not provided, default to 0
    int output_boxcar_width = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-output_boxcar_width") == 0 && i+1 < argc) {
            output_boxcar_width = atoi(argv[i+1]);
        }
    }


    omp_set_num_threads(num_threads);

    int magnitude_array_size;
    float* magnitudes = compute_magnitude(argv[1], &magnitude_array_size);

    if(magnitudes == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    recursive_boxcar_filter(magnitudes, 
        magnitude_array_size, 
        max_boxcar_width, 
        argv[1], 
        candidates_per_boxcar, 
        observation_time_seconds, 
        sigma_threshold, 
        output_boxcar_width);

    profile_candidate_sigma();

    return 0;
}