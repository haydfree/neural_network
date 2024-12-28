#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


typedef struct Node {
    double *input;
    double *coefficients;
    double intercept;
    double output;
    double activation;
    double error;
    double gradient;
    double delta;
    double expected_output;
    int idx;
    int input_size;
    int coefficients_size;
} Node;

typedef struct Layer {
    Node **nodes;
    double *input;
    int num_nodes;
    int idx;
} Layer;

typedef struct Network {
    Layer **layers;
    int num_layers;
} Network;

enum activation_type { STEP, RELU, LEAKY_RELU, SIGMOID, TANH };

Network *network = NULL;

Node *init_node(int input_size, int node_idx, int coefficients_size) {
    int i;

    Node *new_node = malloc(sizeof(Node));

    if (new_node == NULL) {
        printf("Node allocation failed. \n");
        exit(1);
    }

    new_node->input = NULL;
    new_node->coefficients = malloc(sizeof(double) * coefficients_size);

    if (new_node->coefficients == NULL) {
        printf("Coefficients allocation failed. \n");

        free(new_node->coefficients);
        free(new_node->input);
        free(new_node);

        exit(1);
    }

    for (i = 0; i < coefficients_size; i++) {
        new_node->coefficients[i] = 0.0;
    }

    new_node->intercept = 0.0;
    new_node->output = 0.0;
    new_node->activation = 0.0;
    new_node->error = 0.0;
    new_node->gradient = 0.0;
    new_node->delta = 0.0;
    new_node->idx = node_idx;
    new_node->input_size = input_size;
    new_node->coefficients_size = coefficients_size;

    return new_node;
}

Layer *init_layer(int input_size, int number_of_nodes, int layer_idx, int coefficients_size) {
    int node_idx;

    Layer *new_layer = malloc(sizeof(Layer));

    if (new_layer == NULL) {
        printf("Layer allocation failed.\n");
        exit(1);
    }

    new_layer->nodes = malloc(sizeof(Node *) * number_of_nodes);

    if (new_layer->nodes == NULL) {
        printf("Layer nodes allocation failed.\n");
        exit(1);
    }

    new_layer->input = malloc(sizeof(double) * input_size); 

    for (node_idx = 0; node_idx < number_of_nodes; node_idx++) {
        new_layer->nodes[node_idx] = init_node(input_size, node_idx, coefficients_size);
        new_layer->nodes[node_idx]->input = new_layer->input;
    }

    new_layer->num_nodes = number_of_nodes;
    new_layer->idx = layer_idx;

    return new_layer;
}

Network *init_network(int structure[], int number_of_layers) {
    int i, j;

    Network *new_network = malloc(sizeof(Network));

    if (new_network == NULL) {
        printf("Network allocation failed. \n");
        exit(1);
    }

    new_network->layers = malloc(sizeof(Layer *) * number_of_layers);

    if (new_network->layers == NULL) {
        printf("Network layers allocation failed.\n");
        exit(1);
    }

    int coefficients_size;

    for (i = 0; i < number_of_layers; i++) {
        if (i == number_of_layers - 1) {
            coefficients_size = 0;
            new_network->layers[i] = init_layer(structure[i-1], structure[i], i, coefficients_size); 
        } else if (i == 0) {
            coefficients_size = structure[i+1];
            new_network->layers[i] = init_layer(i, structure[i], i, coefficients_size);
        } else {
            coefficients_size = structure[i+1];  
            new_network->layers[i] = init_layer(structure[i-1], structure[i], i, coefficients_size);
        }
    }

    for (i = 1; i < number_of_layers; i++) {
        for (j = 0; j < structure[i-1]; j++) {
            new_network->layers[i]->input[j] = new_network->layers[i-1]->nodes[j]->activation;     
        }
    }

    new_network->num_layers = number_of_layers;

    return new_network;
}

void free_node(Node *node) {
    free(node->coefficients);
    free(node->input);
    free(node);
}

void free_layer(Layer *layer) {
    int i;
    for (i = 0; i < layer->num_nodes; i++) {
        free_node(layer->nodes[i]);
    }
    free(layer->nodes);
    free(layer);
}

void free_network(Network *network) {
    int layer_idx;
    for (layer_idx = 0; layer_idx < network->num_layers; layer_idx++) {
        free_layer(network->layers[layer_idx]);
    }
    free(network->layers);
    free(network);
}

void print_node(Node *node) {
    
    printf("NODE %d: intercept: %.2f, output: %.2f, activation: %.2f, error: %.2f, gradient: %.2f, delta: %.2f\n", 
            node->idx,
            node->intercept,
            node->output,
            node->activation,
            node->error,
            node->gradient,
            node->delta);
    
    int i;
    for (i = 0; i < node->input_size; i++) {
        printf("input %d: %.5f\n", i, node->input[i]);
    }

    for (i = 0; i < node->coefficients_size; i++) {
        printf("coe %d: %.5f\n", i, node->coefficients[i]);
    }

}

void print_layer(Layer *layer) {
    printf("LAYER %d: num_nodes: %d\n", 
            layer->idx, 
            layer->num_nodes);
    int i;
    for (i = 0; i < layer->num_nodes; i++) {
        print_node(layer->nodes[i]);     
    }
}

void print_network(Network *network) {
    int layer_idx;
    for (layer_idx = 0; layer_idx < network->num_layers; layer_idx++) {
        print_layer(network->layers[layer_idx]);
        printf("\n----------\n\n");
    }
}

void print_output(Network *network) {
    printf("OUTPUT: %.5f \n", network->layers[network->num_layers - 1]->nodes[0]->output);
}

double calculate_output(double *inputs, double *coefficients, double intercept, int input_size) {
    double sum = 0.0;
    int i;
    for (i = 0; i < input_size; i++) {
        sum = sum + inputs[i] * coefficients[i];
    }

    sum = sum + intercept;

    return sum;
}

double activation_function(double x, enum activation_type type) {
    switch (type) {
        case STEP:
            return x > 0.0 ? 1.0 : 0.0;
        case RELU:
            return x > 0.0 ? x : 0.0;
        case LEAKY_RELU:
            double leaky_slope = 0.01;
            return x > 0.0 ? x : leaky_slope * x;
        case SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case TANH:
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        default:
            printf("Improper activation function type: %d", type);
            exit(1);
    }
}

double activation_derivative(double x, enum activation_type type) {
    switch (STEP) {
        case STEP:
            return 0.0;
        case RELU:
            return x > 0.0 ? 1.0 : 0.0;
        case LEAKY_RELU:
            double leaky_slope = 0.01;
            return x > 0.0 ? 1.0 : leaky_slope;
        case SIGMOID:
            return x * (1.0 - x);
        case TANH:
            return 1.0 - x * x;
        default:
            printf("Improper activation function type: %d", type);
            exit(1);
    }
}

void give_input(Network *network, double *input) {
    int i;
    for (i = 0; i < network->layers[0]->num_nodes; i++) {
        network->layers[0]->nodes[i]->input = input;
    }
}

void forward_propagation(Network *network) {
    int layer_idx, node_idx;
    for (layer_idx = 0; layer_idx < network->num_layers; layer_idx++) {
        for (node_idx = 0; node_idx < network->layers[layer_idx]->num_nodes; node_idx++) {
            Node *node = network->layers[layer_idx]->nodes[node_idx];
            node->output = calculate_output(
                node->input,
                node->coefficients,
                node->intercept,
                node->input_size
            );
            node->activation = activation_function(node->output, TANH);
        }
    }
}

void back_propagation(Network *network, double learning_rate) {
    int layer_idx, node_idx, weight_idx, next_node_idx;
    for (layer_idx = network->num_layers - 1; layer_idx >= 0; layer_idx--) {
        Layer *layer = network->layers[layer_idx];
        for (node_idx = 0; node_idx < layer->num_nodes; node_idx++) {
            Node *node = layer->nodes[node_idx];
            if (layer_idx == network->num_layers - 1) {
                node->error = node->activation - node->expected_output;
                node->gradient = node->error * activation_derivative(node->activation, TANH);
            } else {
                double error_sum = 0.0;
                for (next_node_idx = 0; next_node_idx < network->layers[layer_idx + 1]->num_nodes; next_node_idx++) {
                    Node *next_node = network->layers[layer_idx + 1]->nodes[next_node_idx];
                    error_sum += next_node->gradient * next_node->coefficients[node_idx];
                }
                node->gradient = error_sum * activation_derivative(node->activation, TANH);
            }
            for (weight_idx = 0; weight_idx < node->input_size; weight_idx++) {
                node->coefficients[weight_idx] -= learning_rate * node->gradient * node->input[weight_idx];
            }
            node->intercept -= learning_rate * node->gradient;
        }
    }
}

void run(Network *network, double *initial_input, double learning_rate, int number_epochs) {
    int i = 0;
    give_input(network, initial_input);
    while (i < number_epochs) {
        forward_propagation(network);
        back_propagation(network, learning_rate);
        print_output(network);
        
        i++;
    }
}

int main() {
    srand((unsigned int)time(NULL));	

    double learning_rate = 0.01;
    double train_set[4][3] = {{0, 0, 0}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}};
    int structure[3] = {2, 2, 1};

    network = init_network(structure, 3);
    run(network, (double *) train_set, learning_rate, 10);

    return 0;
}
