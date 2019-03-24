import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
import os
import gzip
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_path='CIFAR-10'

if __name__ == "__main__":
    
         
    
    
    arguments=sys.argv
    
    if arguments[1]=="--test-data":
      
        #print("11111")
      
#add both the test paths        
        test_path=arguments[2]
        activation='tanh'
        if arguments[4]=="Fashion-MNIST":
      
            #print("qwerty")
            labels_path = os.path.join(test_path,'t10k-labels-idx1-ubyte.gz')
            images_path = os.path.join(test_path,'t10k-images-idx3-ubyte.gz')
            #print(labels_path)
            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)


            images=pd.DataFrame(images)
            labels=pd.DataFrame(labels)
            x_test=(images)
            y_test=(labels)
          

            y_test=np.array(y_test)


            list1=[]
            list2=[]
            for i in range (len(y_test)):
                list1=[0 for i in range(10)]
                list1[y_test[i][0]]=1
                list2.append(list1)
            y_test=list2

            x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
            y_true_cls = tf.argmax(y_true, dimension=1)
            n=1
            activation='tanh'

            def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):

                with tf.variable_scope(name) as scope:
                    shape = [filter_size, filter_size, num_input_channels, num_filters]
                    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name+'_weight')
                    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
                    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                    layer += biases
                    return layer, weights

            def new_pool_layer(input, name):

                with tf.variable_scope(name) as scope:
                    layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    return layer

            def new_relu_layer(input, name):

                with tf.variable_scope(name) as scope:
                    if(str.lower(activation)=='relu'):
                        layer = tf.nn.relu(input)
                    if(str.lower(activation)=='swish'):
                        layer = tf.nn.swish(input)
                    if(str.lower(activation)=='tanh'):
                        layer = tf.nn.tanh(input)
                    if(str.lower(activation)=='sigmoid'):
                        layer = tf.nn.sigmoid(input)
                    return layer

            def new_fc_layer(input, num_inputs, num_outputs, name):

                with tf.variable_scope(name) as scope:
                    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
                    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
                    layer = tf.matmul(input, weights) + biases

                    return layer

            a=[3,5]
            
            for i in range(len(a)):
                if (i==0):
                    layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=n, filter_size=a[i], num_filters=32, name ="conv1")
                    #weight=list(weights_conv1)
                else:
                    layer_conv1, weights_conv1 = new_conv_layer(input=layer_relu1, num_input_channels=32, filter_size=a[i], num_filters=32, name= "conv"+str(i+1))
                    #weight=weight.append(weights_conv1)
                layer_pool1 = new_pool_layer(layer_conv1, name="pool"+str(i+1))
                layer_relu1 = new_relu_layer(layer_pool1, name="relu"+str(i+1))

            #print(layer_relu1)
            num_features = layer_relu1.get_shape()[1:4].num_elements()
            layer_flat = tf.reshape(layer_relu1, [-1, num_features])

            layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

            layer_relu3 = new_relu_layer(layer_fc1, name="relu"+str(i+2))
            layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2") 

            with tf.variable_scope("Softmax"):
                y_pred = tf.nn.softmax(layer_fc2)
                y_pred_cls = tf.argmax(y_pred, dimension=1)

            with tf.name_scope("cross_ent"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
                cost = tf.reduce_mean(cross_entropy)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            writer = tf.summary.FileWriter("Training_FileWriter/")
            writer1 = tf.summary.FileWriter("Validation_FileWriter/")

            tf.summary.scalar('loss', cost)
            tf.summary.scalar('accuracy', accuracy)


            merged_summary = tf.summary.merge_all()
            saver=tf.train.Saver()
            with tf.Session() as sess:

                saver.restore(sess,"mnist.ckpt")
                vali_accuracy,p,q = sess.run([accuracy,y_pred_cls,y_true_cls] , feed_dict={x:x_test, y_true:y_test})
                p=np.array(p)
                out = p
                q=np.array(q)
                out1 = q
                c=f1_score(np.array(out1),np.array(out),average='micro')
                b=f1_score(np.array(out1),np.array(out),average='macro')
                print ("\t- Testing Accuracy:\t{}".format(vali_accuracy))
                print ("\t- Testing F1_micro:\t{}".format(c))
                print ("\t- Testing F1_macro:\t{}".format(b))


            

        elif arguments[4]=="CIFAR-10":
          
            
                        
            def normalize(x):
    
                min_val = np.min(x)
                max_val = np.max(x)
                x = (x-min_val) / (max_val-min_val)
                return x
            
            
            with open(test_path+'/test_batch', 'rb') as fo:

                batch=pickle.load(fo,encoding='latin1')
                features=batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
                labels=batch['labels']
                data6=np.array(features)
                label6=np.array(labels)

         
            
            test_label=np.array(label6)
            
            list1=[]
            list2=[]
            for i in range (len(test_label)):
                list1=[0 for i in range(10)]
                list1[test_label[i]]=1
                list2.append(list1)
            test_label=list2                
                        
            x_test=data6
    
            y_test=test_label
      
            x_test=normalize(x_test)     
      
            x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name='X')
            x_image = tf.reshape(x, [-1, 32, 32, 3])
            y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
            y_true_cls = tf.argmax(y_true, dimension=1)
            n=3
            activatiom='tanh'
            
            def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):

                with tf.variable_scope(name) as scope:
                    shape = [filter_size, filter_size, num_input_channels, num_filters]
                    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name+'_weight')
                    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
                    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                    layer += biases
                    return layer, weights

            def new_pool_layer(input, name):

                with tf.variable_scope(name) as scope:
                    layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    return layer

            def new_relu_layer(input, name):

                with tf.variable_scope(name) as scope:
                    if(str.lower(activation)=='relu'):
                        layer = tf.nn.relu(input)
                    if(str.lower(activation)=='swish'):
                        layer = tf.nn.swish(input)
                    if(str.lower(activation)=='tanh'):
                        layer = tf.nn.tanh(input)
                    if(str.lower(activation)=='sigmoid'):
                        layer = tf.nn.sigmoid(input)
                    return layer

            def new_fc_layer(input, num_inputs, num_outputs, name):

                with tf.variable_scope(name) as scope:
                    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
                    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
                    layer = tf.matmul(input, weights) + biases

                    return layer

            a=[3,5]
            print(len(a))

            for i in range(len(a)):
                if (i==0):
                    layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=n, filter_size=a[i], num_filters=32, name ="conv1")
                    #weight=list(weights_conv1)
                else:
                    layer_conv1, weights_conv1 = new_conv_layer(input=layer_relu1, num_input_channels=32, filter_size=a[i], num_filters=32, name= "conv"+str(i+1))
                    #weight=weight.append(weights_conv1)
                layer_pool1 = new_pool_layer(layer_conv1, name="pool"+str(i+1))
                layer_relu1 = new_relu_layer(layer_pool1, name="relu"+str(i+1))

            #print(layer_relu1)
            num_features = layer_relu1.get_shape()[1:4].num_elements()
            layer_flat = tf.reshape(layer_relu1, [-1, num_features])

            layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

            layer_relu3 = new_relu_layer(layer_fc1, name="relu"+str(i+2))
            layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2") 

            with tf.variable_scope("Softmax"):
                y_pred = tf.nn.softmax(layer_fc2)
                y_pred_cls = tf.argmax(y_pred, dimension=1)

            with tf.name_scope("cross_ent"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
                cost = tf.reduce_mean(cross_entropy)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            writer = tf.summary.FileWriter("Training_FileWriter/")
            writer1 = tf.summary.FileWriter("Validation_FileWriter/")

            tf.summary.scalar('loss', cost)
            tf.summary.scalar('accuracy', accuracy)


            merged_summary = tf.summary.merge_all()
            
            
            saver=tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess,'cifar.ckpt')
                vali_accuracy,p,q = sess.run([accuracy,y_pred_cls,y_true_cls] , feed_dict={x:x_test, y_true:y_test})
                p=np.array(p)
                #vali_accuracy=sess.run(accuracy,feed_dict={x:x_test,y_true:y_test})
                out = p
                q=np.array(q)
                out1 = q
                c=f1_score(np.array(out1),np.array(out),average='micro')
                b=f1_score(np.array(out1),np.array(out),average='macro')
                print ("\t- Testing Accuracy:\t{}".format(vali_accuracy))
                print ("\t- Testing F1_micro:\t{}".format(c))
                print ("\t- Testing F1_macro:\t{}".format(b))
    elif arguments[1]=="--train-data":
     
        train_path=arguments[2]
        test_path=arguments[4]
        
        list1=arguments[8]
        k=9
        i=1
        list2=[]
        list2.append(int(list1[1:]))
        while(1):
            if arguments[k][-1]==']':
                list2.append(int(arguments[k][:-1]))
                i+=1
                break
            list2.append(int(arguments[k]))
            i+=1
            k+=1

       
        list_config=list2
        activation=arguments[k+2]
        
        #print(list1)
        
        
        if arguments[6]=="Fashion-MNIST":
            path1=train_path 
            
            labels_path = os.path.join(path1,'train-labels-idx1-ubyte.gz')
            images_path = os.path.join(path1,'train-images-idx3-ubyte.gz')
            #print(labels_path)
            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)
                
            images=pd.DataFrame(images)
            labels=pd.DataFrame(labels)
            train_set=(images)
            train_label=(labels)
            train_label=np.array(train_label)
            list1=[]
            list2=[]
            for i in range (len(train_label)):
                list1=[0 for j in range(10)]
                list1[train_label[i][0]]=1
                list2.append(list1)
            train_label=list2
            labels_path1 = os.path.join(test_path,'t10k-labels-idx1-ubyte.gz')
            images_path1 = os.path.join(test_path,'t10k-images-idx3-ubyte.gz')
    
    
            
            with gzip.open(labels_path1, 'rb') as lbpath:
                labels1 = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path1, 'rb') as imgpath:
                images1 = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels1), 784)
            
            
            
            
            images1=pd.DataFrame(images1)
            labels1=pd.DataFrame(labels1)
            test_set=(images1)
            test_label=(labels1)
            
            test_label=np.array(test_label)
            
            
            
            list3=[]
            list4=[]
            for i in range (len(test_label)):
                list3=[0 for j in range(10)]
                list3[test_label[i][0]]=1
                list4.append(list3)
            test_label=list4
            #print(test_label[0:10])
            
            x_train=train_set
            x_test=test_set
            y_train=train_label
            y_test=test_label
            
     
            
            #x_train, x_test, y_train, y_test = train_test_split(train_set,train_label, test_size=0.20, random_state=42)

            x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
            y_true_cls = tf.argmax(y_true, dimension=1)
            n=1
            
        elif arguments[6]=="CIFAR-10":
            def normalize(x):
    
                min_val = np.min(x)
                max_val = np.max(x)
                x = (x-min_val) / (max_val-min_val)
                return x
            
          
            with open(train_path + '/data_batch_1', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data1=np.array(features)
                label1=np.array(labels)         

            with open(train_path + '/data_batch_2', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data2=np.array(features)
                label2=np.array(labels)  
          
            with open(train_path + '/data_batch_3', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data3=np.array(features)
                label3=np.array(labels)    
                
            with open(train_path + '/data_batch_4', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data4=np.array(features)
                label4=np.array(labels)  
          
            with open(train_path + '/data_batch_5', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data5=np.array(features)
                label5=np.array(labels) 
            test_path=""
             
            with open(test_path + '/test_batch', mode='rb') as file:
        
                batch = pickle.load(file, encoding='latin1')        
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                data6=np.array(features)
                label6=np.array(labels) 
          
            train_label=np.concatenate((label1, label2,label3,label4,label5), axis=None)
            test_label=label6
            train_set=np.concatenate((data1, data2, data3, data4, data5), axis=0)
            test_set=data6
            train_set=normalize(train_set)
            test_set=normalize(test_set)
          
            train_label=np.array(train_label)
            test_label=np.array(test_label)
            list1=[]
            list2=[]
            for i in range (len(train_label)):
                list1=[0 for i in range(10)]
                list1[train_label[i]]=1
                list2.append(list1)
            train_label=list2
            
            list1=[]
            list2=[]
            for i in range (len(test_label)):
                list1=[0 for i in range(10)]
                list1[test_label[i]]=1
                list2.append(list1)
            test_label=list2
                        
            x_train=train_set
            x_test=test_set
            y_train=train_label
            y_test=test_label
            #x_train, x_test, y_train, y_test = train_test_split(train_set,train_label, test_size=0.20, random_state=42)
            x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name='X')
            x_image = tf.reshape(x, [-1, 32, 32, 3])
            y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
            y_true_cls = tf.argmax(y_true, dimension=1)
            n=3
          
        def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
            with tf.variable_scope(name) as scope:
                shape = [filter_size, filter_size, num_input_channels, num_filters]
                weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name+'_weight')
                biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
                layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                layer += biases
                return layer, weights
            
        def new_pool_layer(input, name):
    
            with tf.variable_scope(name) as scope:
                layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                return layer
            
        def new_relu_layer(input, name):
    
            with tf.variable_scope(name) as scope:
                if(str.lower(activation)=='relu'):
                    layer = tf.nn.relu(input)
                if(str.lower(activation)=='swish'):
                    layer = tf.nn.swish(input)
                if(str.lower(activation)=='tanh'):
                    layer = tf.nn.tanh(input)
                if(str.lower(activation)=='sigmoid'):
                    layer = tf.nn.sigmoid(input)
                return layer
            
        def new_fc_layer(input, num_inputs, num_outputs, name):
    
            with tf.variable_scope(name) as scope:
                weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
                biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
                layer = tf.matmul(input, weights) + biases

                return layer

        a=list_config
                
        for i in range(len(a)):
            if (i==0):
                layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=n, filter_size=a[i], num_filters=32, name ="conv1")
            else:
                layer_conv1, weights_conv1 = new_conv_layer(input=layer_relu1, num_input_channels=32, filter_size=a[i], num_filters=32, name= "conv"+str(i+1))
            layer_pool1 = new_pool_layer(layer_conv1, name="pool"+str(i+1))
            layer_relu1 = new_relu_layer(layer_pool1, name="relu"+str(i+1))

        num_features = layer_relu1.get_shape()[1:4].num_elements()
        layer_flat = tf.reshape(layer_relu1, [-1, num_features])

        layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

        layer_relu3 = new_relu_layer(layer_fc1, name="relu"+str(i+2))
        layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2") 
        
        with tf.variable_scope("Softmax"):
            y_pred = tf.nn.softmax(layer_fc2)
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            
        with tf.name_scope("cross_ent"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
            cost = tf.reduce_mean(cross_entropy)
            
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
            
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        writer = tf.summary.FileWriter("Training_FileWriter/")
        writer1 = tf.summary.FileWriter("Validation_FileWriter/")
        
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)


        merged_summary = tf.summary.merge_all()
        
        
        
        num_epochs = 1
        batch_size = 100 
        
        
       
        with tf.Session() as sess:
    
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            for epoch in range(num_epochs):

              
                train_accuracy = 0
                k=0
                out =[]
                out1=[]
                for batch in range(0, int(len(y_train)/batch_size)):
                    #print(i)
                    #print((len(y_train)/batch_size))
                    x_batch  = x_train[k:k+batch_size]
                    y_true_batch = y_train[k:k+batch_size]
                    k=k+batch_size
                    #print(len(y_train))
                    #print(batch_size)
                    feed_dict_train = {x: x_batch, y_true: y_true_batch}


                    sess.run(optimizer, feed_dict=feed_dict_train)


                    train_accurac,y,z = sess.run([accuracy,y_pred_cls,y_true_cls], feed_dict=feed_dict_train)
                    train_accuracy+=train_accurac
                    y=np.array(y)
                    out = np.concatenate((out , y), axis=None)
                    z=np.array(z)
                    out1 = np.concatenate((out1 , z), axis=None)
                train_accuracy /= int(len(y_train)/batch_size)
                end_time = time.time()
                print("Epoch "+str(epoch+1)+" completed")
                print ("\t- Training Accuracy:\t{}".format(train_accuracy))
                c=f1_score(np.array(out1),np.array(out),average='micro')
                b=f1_score(np.array(out1),np.array(out),average='macro')
                print ("\t- F1_micro:\t{}".format(c))
                print ("\t- F1_macro:\t{}".format(b))
                vali_accuracy = sess.run(accuracy , feed_dict={x:x_test, y_true:y_test})
           
              
            vali_accuracy,p,q = sess.run([accuracy,y_pred_cls,y_true_cls] , feed_dict={x:x_test, y_true:y_test})
            p=np.array(p)
            out = p
            q=np.array(q)
            out1 = q
            c=f1_score(np.array(out1),np.array(out),average='micro')
            b=f1_score(np.array(out1),np.array(out),average='macro')
            print ("\t- Testing Accuracy:\t{}".format(vali_accuracy))
            print ("\t- Testing F1_micro:\t{}".format(c))
            print ("\t- Testing F1_macro:\t{}".format(b))
