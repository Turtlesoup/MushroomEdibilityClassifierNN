import os
import tensorflow as tf
import csv

#############################################################
##	       Create Training/Test Data sets from csv         ##
#############################################################

train_data = []
train_classes = []
test_data = []
test_classes = []

f = open('mushrooms_train_data.csv', 'r')
reader = csv.reader(f)
for row in reader:
	train_data.append(list(map(int, row)))

f = open('mushrooms_train_classes.csv', 'r')
reader = csv.reader(f)
for row in reader:
	train_classes.append(list(map(int, row)))

f = open('mushrooms_test_data.csv', 'r')
reader = csv.reader(f)
for row in reader:
	test_data.append(list(map(int, row)))

f = open('mushrooms_test_classes.csv', 'r')
reader = csv.reader(f)
for row in reader:
	test_classes.append(list(map(int, row)))

#########################################
##				Training               ##
#########################################

#sess = tf.InteractiveSession()

# Parameters
learning_rate = 0.001

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 22 # Mushroom data input (22 different parameters: hight, cap size, spot colour, etc)
n_classes = 2 # 2 class types (edible/poisonous)
n_trainingSets = len(train_data) # Set to len(train_data) to train on all data

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

print("Starting training on", n_trainingSets, "sets")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Loop over all training data
    i = 0
    
    while i < n_trainingSets:
        batch_x = tf.reshape(train_data[i], [-1, 22]).eval()
        batch_y = tf.reshape(train_classes[i], [-1, 2]).eval()

        print(i+1, "/", n_trainingSets)
        i+=1;

        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y})

    print("Optimization Finished!")

    ########################################
	##				Testing               ##
	########################################
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_data, y: test_classes}))