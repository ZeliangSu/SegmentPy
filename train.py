import tensorflow as tf


def test_train(train_op, loss_op, accuracy):
    with tf.Session() as sess:
        # ... init our variables, ...
        sess.run(tf.global_variables_initializer())

        # ... add the coordinator, ...
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # ... check the accuracy before training (without feed_dict!), ...
        sess.run(accuracy)

        # ... train ...
        for i in range(5000):
            #  ... without sampling from Python and without a feed_dict !
            _, loss = sess.run([train_op, loss_op])

            # We regularly check the loss
            if i % 500 == 0:
                print('iter:%d - loss:%f' % (i, loss))

        # Finally, we check our final accuracy
        sess.run(accuracy)

        coord.request_stop()
        coord.join(threads)


def MBGD_segnet():
    pass

