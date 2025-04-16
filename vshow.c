#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

// Base64 encoding table
static const char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Function to encode data to base64
size_t base64_encode(const unsigned char *data, size_t input_length, char *encoded_data) {
    size_t output_length = 4 * ((input_length + 2) / 3);
    size_t i, j;
    for (i = 0, j = 0; i < input_length;) {
        uint32_t octet_a = i < input_length ? data[i++] : 0;
        uint32_t octet_b = i < input_length ? data[i++] : 0;
        uint32_t octet_c = i < input_length ? data[i++] : 0;
        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        encoded_data[j++] = base64_table[(triple >> 18) & 0x3F];
        encoded_data[j++] = base64_table[(triple >> 12) & 0x3F];
        encoded_data[j++] = base64_table[(triple >> 6) & 0x3F];
        encoded_data[j++] = base64_table[triple & 0x3F];
    }
    // Add padding if needed
    size_t mod_table[] = {0, 2, 1};
    for (i = 0; i < mod_table[input_length % 3]; i++)
        encoded_data[output_length - 1 - i] = '=';
    return output_length;
}

int main() {
    // Initialize random seed for image ID
    srand(time(NULL));

    // Read vector from stdin
    double *vector = NULL;
    size_t vector_size = 0;
    size_t vector_capacity = 0;
    char line[1024];

    // Read one float per line from stdin
    while (fgets(line, sizeof(line), stdin) != NULL) {
        // Resize if needed
        if (vector_size >= vector_capacity) {
            vector_capacity = vector_capacity ? vector_capacity * 2 : 8;
            double *new_vector = realloc(vector, vector_capacity * sizeof(double));
            if (!new_vector) {
                fprintf(stderr, "Memory allocation failed\n");
                free(vector);
                return 1;
            }
            vector = new_vector;
        }

        // Convert to double and add to vector
        vector[vector_size++] = atof(line);
    }

    // Check if we got any values
    if (vector_size == 0) {
        fprintf(stderr, "No valid vector values found\n");
        free(vector);
        return 1;
    }

    // Find max absolute value for normalization
    double max_abs_val = 0.0;
    for (size_t i = 0; i < vector_size; i++) {
        double abs_val = fabs(vector[i]);
        if (abs_val > max_abs_val) {
            max_abs_val = abs_val;
        }
    }

    // Normalize the vector
    if (max_abs_val > 0.0) {
        for (size_t i = 0; i < vector_size; i++) {
            vector[i] /= max_abs_val;
        }
    }

    // Calculate square dimension
    size_t square_dim = 1;
    while (square_dim * square_dim < vector_size) {
        square_dim++;
    }

    // Add 2 pixels total for border (1 on each side)
    size_t image_dim = square_dim + 2;

    // Create the bitmap (RGB)
    unsigned char *bitmap = malloc(image_dim * image_dim * 3);
    if (!bitmap) {
        fprintf(stderr, "Memory allocation failed\n");
        free(vector);
        return 1;
    }

    // Fill with border color (gray #999999)
    memset(bitmap, 0x99, image_dim * image_dim * 3);

    // Fill the inner square with vector components
    for (size_t y = 0; y < square_dim; y++) {
        for (size_t x = 0; x < square_dim; x++) {
            size_t idx = y * square_dim + x;
            size_t pos = ((y + 1) * image_dim + (x + 1)) * 3;  // Offset by 1 for border

            if (idx < vector_size) {
                double val = vector[idx];
                if (val > 0) {
                    // Positive: Red (scaling from black to full red)
                    bitmap[pos] = (unsigned char)(val * 255);  // R
                    bitmap[pos + 1] = 0;                       // G
                    bitmap[pos + 2] = 0;                       // B
                } else if (val < 0) {
                    // Negative: Green (scaling from black to full green)
                    bitmap[pos] = 0;                           // R
                    bitmap[pos + 1] = (unsigned char)(-val * 255); // G
                    bitmap[pos + 2] = 0;                       // B
                } else {
                    // Zero: Black
                    bitmap[pos] = 0;      // R
                    bitmap[pos + 1] = 0;  // G
                    bitmap[pos + 2] = 0;  // B
                }
            } else {
                // Fill any unused cells with black
                bitmap[pos] = 0;      // R
                bitmap[pos + 1] = 0;  // G
                bitmap[pos + 2] = 0;  // B
            }
        }
    }

    // Setup base64 encoding
    size_t bitmap_size = image_dim * image_dim * 3;
    size_t encoded_size = 4 * ((bitmap_size + 2) / 3);
    char *encoded_data = malloc(encoded_size + 1);
    if (!encoded_data) {
        fprintf(stderr, "Memory allocation failed\n");
        free(vector);
        free(bitmap);
        return 1;
    }

    // Random ID for the image
    long id = rand();

    // Encode and display
    base64_encode(bitmap, bitmap_size, encoded_data);
    encoded_data[encoded_size] = '\0';

    printf("\033_Ga=T,i=%lu,f=24,s=%zu,v=%zu,c=20,r=10,q=2;", id, image_dim, image_dim);
    printf("%s", encoded_data);
    printf("\033\\");
    printf("\n");
    fflush(stdout);

    // Clean up
    free(encoded_data);
    free(bitmap);
    free(vector);

    return 0;
}
