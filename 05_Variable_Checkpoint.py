# -*- coding: UTF-8 -*-

import tensorflow as tf

# Check Checkpoint

### Using 1) tf.train.get_checkpoint_state(saved_dir_path)
# ckpt_state = tf.train.get_checkpoint_state("saved")
# print(type(ckpt_state))                                     # <class 'tensorflow.python.training.checkpoint_state_pb2.CheckpointState'>
# # saved\tf-basic03-tensor-concept05
# print("첫번째 정보 사용법:", ckpt_state.model_checkpoint_path)
# # ['saved\\tf-basic03-tensor-concept05-0', 'saved\\tf-basic03-tensor-concept05-1', 'saved\\tf-basic03-tensor-concept05-2', 'saved\\tf-basic03-tensor-concept05']
# print("두번째 정보 사용법:", ckpt_state.all_model_checkpoint_paths)

### Using 2) tf.train.latest_checkpoint(saved_dir_path)
recent_ckpt_job_path = tf.train.latest_checkpoint("saved")
print("Checkpoint Path:", recent_ckpt_job_path)               # saved\tf-basic03-tensor-concept05


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
    
    # Restore Checkpoint
    if recent_ckpt_job_path != None:
        ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("saved"))
    
    # Print the initial value of 'state'
    print("Initial Value:", sess.run(state))
    
    for lvalue in range(3):
        sess.run(update)
        print("Running Value:", sess.run(state))
#         ckpt_path = saver.save(sess, "saved/tf-basic03-tensor-concept05", lvalue)
    
    # Save Checkpoint
    ckpt_path = saver.save(sess, "saved/tf-basic03-tensor-concept05")
    
# Graph 의 실행 상태를 저장하는 것이 Variables 의 특징이다
# 위 예제에서 Variables 는 간단한 counter 의 역할을 하게 된다.
# assign() operation 은 add()와 같은 graph 상의 하나의 operation 이다
# 당옇니 session.run() 하기전까지는 실행되지 않는다
# Neural network 에서 tensor 에 weight 값을 저장할 때 Variables 을 사용하게 된다.


### tf.train.Saver
# 1)
# Checkpoint Path: None
# Initial Value: 0
# Running Value: 1
# Running Value: 2
# Running Value: 3
# 2)
# Checkpoint Path: saved\tf-basic03-tensor-concept05
# Initial Value: 3
# Running Value: 4
# Running Value: 5
# Running Value: 6
# 3)
# Checkpoint Path: saved\tf-basic03-tensor-concept05
# Initial Value: 6
# Running Value: 7
# Running Value: 8
# Running Value: 9
