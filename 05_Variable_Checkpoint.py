# -*- coding: UTF-8 -*-

import tensorflow as tf

# Two different method using Check Checkpoint

### Using 1) tf.train.get_checkpoint_state(saved_dir_path)
# ckpt_state = tf.train.get_checkpoint_state("saved")
# print(type(ckpt_state))                                     # <class 'tensorflow.python.training.checkpoint_state_pb2.CheckpointState'>

# # saved\05_Variable_Checkpoint
# print("1.1) Save Model:", ckpt_state.model_checkpoint_path)

# # ['saved\\05_Variable_Checkpoint-0', 'saved\\05_Variable_Checkpoint-1', 'saved\\05_Variable_Checkpoint-2', ...]
# print("1.2) Save All Model:", ckpt_state.all_model_checkpoint_paths)

### Using 2) tf.train.latest_checkpoint(saved_dir_path)
recent_ckpt_job_path = tf.train.latest_checkpoint("saved")
print("Checkpoint Path:", recent_ckpt_job_path)               # saved\05_Variable_Checkpoint


# Create a Variable, that will be initialized to the scalar value 0
state = tf.Variable(0, name="Counter")

# Create an Op to add one to 'state'
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Saver
saver = tf.train.Saver()

# Variables must be initialized by running an 'init' op 
# after having launched the graph.
# We first have to add the 'init' Op the the graph.

init_op = tf.global_variables_initializer()

# Launch the graph and run the ops
with tf.Session() as sess:
    
    # Run the 'init' op
    sess.run(init_op)
    
    # Restore Checkpoint if Checkpoint is available
    if recent_ckpt_job_path != None:
        ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("saved"))
    
    # Print the initial value of 'state'
    print("Initial Value:", sess.run(state))
    
    for lvalue in range(3):
        sess.run(update)
        print("Running Value:", sess.run(state))
    
    # Save Checkpoint
    ckpt_path = saver.save(sess, "saved/05_Variable_Checkpoint")
    
### 1) 1st Run
# Checkpoint Path: None
# Initial Value: 0
# Running Value: 1
# Running Value: 2
# Running Value: 3
### 2) 2nd Run
# Checkpoint Path: saved\05_Variable_Checkpoint
# Initial Value: 3
# Running Value: 4
# Running Value: 5
# Running Value: 6
### 3) 3rd Run
# Checkpoint Path: saved\05_Variable_Checkpoint
# Initial Value: 6
# Running Value: 7
# Running Value: 8
# Running Value: 9
