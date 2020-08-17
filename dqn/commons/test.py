# Initialize input 
def input_initialization(env_info,img_size=80,Num_colorChannel=1,Num_skipFrame=4,Num_stackFrame=4,Num_dataSize= 366):
    #Informations d'observation vectorielle
    #Dans ce simulateur, la taille de l'observation vectorielle est de 373 .
    #0 ~ 359: Données LIDAR (1 particule pour 1 degré)
    #360 ~ 362: avertissement gauche, avertissement droit, avertissement avant (0: faux, 1: vrai)
    #363: distance avant normalisée
    #364: Vitesse du véhicule en marche avant
    #365: Vitesse du véhicule hôte
    #0 ~ 365 sont utilisés comme données d'entrée pour le capteur
    #366 ~ 372 sont utilisés pour envoyer des informations
    #366: Nombre de dépassements dans un épisode
    #367: Nombre de changement de voie dans un épisode
    #368 ~ 372: récompense longitudinale, récompense latérale, 
    #   récompense de dépassement, récompense de violation, récompense de collision

    # Examine the observation space for the default brain
    Num_obs = len(env_info.visual_observations)
    
    # Observation
    observation_stack_obs = np.zeros([img_size, img_size, Num_colorChannel * Num_obs])
    
    for i in range(Num_obs):
        observation = 255 * env_info.visual_observations[i]
        observation = np.uint8(observation)
        observation = np.reshape(observation, (observation.shape[1], observation.shape[2], 3))
        observation = cv2.resize(observation, (img_size, img_size))

        if Num_colorChannel == 1:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = np.reshape(observation, (img_size, img_size))

        if Num_colorChannel == 3:
            observation_stack_obs[:,:, Num_colorChannel * i: Num_colorChannel * (i+1)] = observation
        else:
            observation_stack_obs[:,:, i] = observation

    observation_set = []

    # State
    state = env_info.vector_observations[0][:-7]
    state_set = []
        
    for i in range(Num_skipFrame * Num_stackFrame):
        observation_set.append(observation_stack_obs)
        state_set.append(state)
    
    # Stack the frame according to the number of skipping and stacking frames using observation set
    observation_stack = np.zeros((img_size, img_size, Num_colorChannel * Num_stackFrame * Num_obs))
    state_stack = np.zeros((Num_stackFrame, Num_dataSize))
    
    for stack_frame in range(Num_stackFrame):
        observation_stack[:,:,Num_obs * stack_frame: Num_obs * (stack_frame+1)] = observation_set[-1 - (Num_skipFrame * stack_frame)]
        state_stack[(Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (Num_skipFrame * stack_frame)]
    
    observation_stack = np.uint8(observation_stack)
    state_stack = np.uint8(state_stack)
    
    return observation_stack, observation_set, state_stack, state_set

# Resize input information 
def resize_input(env_info, observation_set, state_set,img_size=80,Num_colorChannel=1,Num_skipFrame=4,Num_stackFrame=4,Num_dataSize= 366):
    # Stack observation according to the number of observations
    observation_stack_obs = np.zeros([img_size, img_size, Num_colorChannel * Num_obs])

    for i in range(Num_obs):
        observation = 255 * env_info.visual_observations[i]
        observation = np.uint8(observation)
        observation = np.reshape(observation, (observation.shape[1], observation.shape[2], 3))
        observation = cv2.resize(observation, (img_size, img_size))
        
        if Num_colorChannel == 1:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = np.reshape(observation, (img_size, img_size))

        if Num_colorChannel == 3:
            observation_stack_obs[:,:, Num_colorChannel * i: Num_colorChannel * (i+1)] = observation
        else:
            observation_stack_obs[:,:,i] = observation
    
    # Add observations to the observation_set
    observation_set.append(observation_stack_obs)
    
    # State 
    state = env_info.vector_observations[0][:-7]

    # Add state to the state_set
    state_set.append(state)
    
    # Stack the frame according to the number of skipping and stacking frames using observation set
    observation_stack = np.zeros((img_size, img_size, Num_colorChannel * Num_stackFrame * Num_obs))
    state_stack = np.zeros((Num_stackFrame, Num_dataSize))

    for stack_frame in range(Num_stackFrame):
        observation_stack[:,:,Num_obs * stack_frame: Num_obs * (stack_frame+1)] = observation_set[-1 - (Num_skipFrame * stack_frame)]
        state_stack[(Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (Num_skipFrame * stack_frame)]

    del observation_set[0]
    del state_set[0]
    
    observation_stack = np.uint8(observation_stack)
    state_stack = np.uint8(state_stack)
        
    return observation_stack, observation_set, state_stack, state_set

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

# Xavier Weights initializer
def xavier_initializer(shape):
	dim_sum = np.sum(shape)
	if len(shape) == 1:
		dim_sum += 1
	bound = np.sqrt(2.0 / dim_sum)
	return tf.random_uniform(shape, minval=-bound, maxval=bound)

# Convolution function
def conv2d(x,w, stride):
	return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

first_conv   = [8,8,Num_colorChannel * Num_stackFrame * Num_obs,32]
second_conv  = [4,4,32,64]
third_conv   = [3,3,64,64]
first_dense  = [10*10*64 + Num_cellState, 512]
second_dense = [first_dense[1], Num_action]

def build_network(first_conv,second_conv,first_dense,num_cellState=512):
    '''
    first_conv   = [8,8,Num_colorChannel * Num_stackFrame * Num_obs,32]
    second_conv  = [4,4,32,64]
    third_conv   = [3,3,64,64]
    first_dense  = [10*10*64 + Num_cellState, 512]
    second_dense = [first_dense[1], Num_action]
    '''

    # Input
    #x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, Num_colorChannel * Num_stackFrame * Num_obs])
    

    #x_sensor = tf.placeholder(tf.float32, shape = [None, Num_stackFrame, Num_dataSize])
    
    
    def network_fn(input_placeholder):
        '''
        input_placeholder=[tf_image,tf_sensor]
        '''

        x_normalize = (input_placeholder[0]- (255.0/2)) / (255.0/2)
        x_unstack = tf.unstack(input_placeholder[1], axis = 1)

        # Convolution variables
        w_conv1 = weight_variable(first_conv)
        b_conv1 = bias_variable([first_conv[3]])
        
        w_conv2 = weight_variable(second_conv)
        b_conv2 = bias_variable([second_conv[3]])
        w_conv3 = weight_variable(third_conv)
        b_conv3 = bias_variable([third_conv[3]])
        
        # Densely connect layer variables
        w_fc1 = weight_variable(first_dense)
        b_fc1 = bias_variable([first_dense[1]])

        #w_fc2 = weight_variable(second_dense)
        #b_fc2 = bias_variable([second_dense[1]])
        # LSTM cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = num_cellState)            
        rnn_out, rnn_state = tf.nn.static_rnn(inputs = x_unstack, cell = cell, dtype = tf.float32)

        rnn_out=tf.contrib.layers.layer_norm(rnn_out[-1], center=True, scale=True)
        
        # Network
        h_conv1 = tf.contrib.layers.layer_norm(tf.nn.relu(conv2d(x_normalize, w_conv1, 4) + b_conv1),center=True, scale=True)
        h_conv2 = tf.contrib.layers.layer_norm(tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2),center=True, scale=True)
        h_conv3 = tf.contrib.layers.layer_norm(tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3),center=True, scale=True)
        h_pool3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64])

        #rnn_out = rnn_out[-1]
        h_concat = tf.concat([h_pool3_flat, rnn_out], axis = 1)
        output = tf.nn.relu(tf.matmul(h_concat, w_fc1)+b_fc1)

        output =tf.contrib.layers.layer_norm(output, center=True, scale=True)

        #output = tf.matmul(h_fc1,  w_fc2)+b_fc2

        return output

    return network_fn