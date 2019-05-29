import tensorflow as tf

restorer = tf.train.import_meta_graph(
    './dummy/ckpt/step5/ckpt.meta',
        # input_map={
        #     'input_pipeline/input_cond/Merge_1': input
        # },
        clear_devices=True
    )

with tf.Session() as sess:
    restorer.restore(sess, './dummy/ckpt/step5/ckpt')
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print([v.name for v in all_vars])
    ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0')]
    bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0')]
    print('ws shape:', [elt.shape for elt in ws])
    print('bs shape:', [elt.shape for elt in bs])
    print('ws:', ws)
    print('bs:', bs)
