from generate import *
from param import *
# The discriminator network based on the CNN classifier
def discriminator_2(x_onehot,encoder_state, batch_size, seq_len, vocab_size,_embed_ph,D_with_state = param.D_with_state,filter_dim = 100):
    # get the embedding dimension for each presentation
    #assert isinstance(emb_dim_single, int) and emb_dim_single > 0

    filter_sizes = [2, 3, 4, 5]
    num_filters = [300, 300, 300, 300]
    dropout_keep_prob = 0.75
    num_rep = 64
    dis_emb_dim = param.INPUT_DIM
    emb_dim_single = int(dis_emb_dim / num_rep)
    d_embeddings = tf.get_variable(name='d_emb', shape=[vocab_size,dis_emb_dim],
                                   initializer=xavier_initializer())#changed from trainable = False
#    _embed_init = d_embeddings.assign(_embed_ph)
    #the embeddings of the discriminator is different with the one for generator
    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    emb_x_re = tf.matmul(input_x_re, d_embeddings)
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim

    emb_x_expanded = tf.expand_dims(emb_x, -1)  # batch_size x seq_len x dis_emb_dim x 1
    print('shape of emb_x_expanded: {}'.format(emb_x_expanded.get_shape().as_list()))

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for filter_size, num_filter in zip(filter_sizes, num_filters):
        conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
                      d_h=1, d_w=emb_dim_single, padding='VALID'#d is stride, k is kernel size
                      ,scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+1) x num_rep x num_filter
        out = tf.nn.relu(conv, name="relu")
        # pooled = tf.nn.max_pool(out, ksize=[1, seq_len - filter_size + 1, 1, 1],
        #                         strides=[1, 1, 1, 1], padding='VALID',
        #                         name="pool")# batch_size x 1 x num_rep x num_filter
        pooled = tf.reduce_max(out,axis=1)
        pooled = tf.reshape(pooled,[BATCH_SIZE,1,num_rep,num_filter])
        pooled_outputs.append(pooled)

    # Combine all the pooled features
    if D_with_state:
        num_filters_total = sum(num_filters)
    else:
        num_filters_total = sum(num_filters)
    h_pool = tf.concat(pooled_outputs, 3)  # batch_size x 1 x num_rep x num_filters_total

    print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    if D_with_state:
        tile_state = tf.reshape(tf.tile(encoder_state, [1, num_rep, 1]), [BATCH_SIZE*num_rep, int(encoder_state.shape[-1])])
        state = linear(tile_state, output_size=num_filters_total, use_bias=True, scope='state_fc')
        h_pool_flat = tf.multiply(state,h_pool_flat)


    # Add highway
    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)  # (batch_size*num_rep) x num_filters_total

    # Add dropout
    # if D_with_state:
    #     h_highway = tf.concat([tf.reshape(tf.tile(encoder_state, [1, num_rep, 1]), [-1, encoder_state.shape[-1]]), h_highway], 1)
    print('shape of h_highway: {}'.format(h_highway.get_shape().as_list()))
    h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout')

    # fc
    fc_out = linear(h_drop, output_size=100, use_bias=True, scope='fc')
    logits = linear(fc_out, output_size=1, use_bias=True, scope='logits')
    logits = tf.squeeze(logits, -1)  # batch_size*num_rep

    return logits,h_highway