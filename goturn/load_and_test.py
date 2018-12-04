import logging
import time
import tensorflow as tf
import os,sys
import goturn_net
#import goturn_net
import cv2
from PIL import Image
NUM_EPOCHS = 500
BATCH_SIZE = 10
WIDTH = 227
HEIGHT = 227
logfile = "test.log"
test_txt = "ou_train.txt"
def load_train_test_set(train_file):
    '''
    return train_set or test_set
    example line in the file:
    <target_image_path>,<search_image_path>,<x1>,<y1>,<x2>,<y2>
    (<x1>,<y1>,<x2>,<y2> all relative to search image)
    '''
    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    train_target = []
    train_search = []
    train_box = []
    for line in trainlines:
        #print(line)
        line = line.split(",")
        # remove too extreme cases
        # if (float(line[2]) < -0.3 or float(line[3]) < -0.3 or float(line[4]) > 1.2 or float(line[5]) > 1.2):
        #     continue
        train_target.append(line[0])
        train_search.append(line[1])
        box = [10*float(line[2]), 10*float(line[3]), 10*float(line[4]), 10*float(line[5])]
       
        train_box.append(box)
#         img = cv2.imread(line[1],1)
#         height, width, channels = img.shape 
#         gt1=(int(width*float(line[2])),int(height*float(line[3])))
#         gt2=(int(width*float(line[4])),int(height*float(line[5])))
#         print(height)
#         print(width)
#         print(gt1)
#         print(gt2)
#         print(float(line[2])*width)
#         print(float(line[3])*height)
#         print(float(line[4])*width)
#         print(float(line[5])*height)
#         cv2.rectangle(img, gt1, gt2, (0,255,0), 2, 1)
#         #cv2.line(img,(23,88),(23+66,88+55),(255,0,0),5)
#         cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
#         #imS = cv2.resize(img, (1280, 720))                    # Resize image
#         cv2.imshow("output", img)                            # Show image
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         print("target"+line[0])
#         print("search"+line[1])
#         print("bbox1"+line[2])
#         print("bbox2"+line[3])
    ftrain.close()
    print("len:%d"%(len(train_target)))
    
    return [train_target, train_search, train_box]

def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    search_img = tf.read_file(input_queue[0])
    target_img = tf.read_file(input_queue[1])
    search_path=input_queue[0]
    target_path=input_queue[1]
    search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels = 3))
    search_tensor = tf.image.resize_images(search_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels = 3))
    target_tensor = tf.image.resize_images(target_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    box_tensor = input_queue[2]
    return [search_tensor, target_tensor, box_tensor,search_path,target_path]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor, box_tensor,search_path,target_path] = data_reader(input_queue)
    [search_batch, target_batch, box_batch] = tf.train.batch(
        [search_tensor, target_tensor, box_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)
    return [search_batch, target_batch, box_batch,search_path,target_path]


if __name__ == "__main__":
    if (os.path.isfile(logfile)):
        os.remove(logfile)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
        level=logging.DEBUG,filename=logfile)
    print("loading")
    [train_target, train_search, train_box] = load_train_test_set(test_txt)
    target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
    search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
    box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
    input_queue = tf.train.slice_input_producer([search_tensors, target_tensors, box_tensors],shuffle=False)
    batch_queue = next_batch(input_queue)
    tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = False)
    tracknet.build()
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt_dir = "./orcheckpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    try:
        for i in range(0, int(len(train_box)/BATCH_SIZE)):
            cur_batch = sess.run(batch_queue)
            start_time = time.time()
            [batch_loss, fc4] = sess.run([tracknet.loss, tracknet.fc4],feed_dict={tracknet.image:cur_batch[0],
                    tracknet.target:cur_batch[1], tracknet.bbox:cur_batch[2]})
            logging.info('batch box: %s' %(fc4))
            logging.info('gt batch box: %s' %(cur_batch[2]))
            logging.info('batch loss = %f'%(batch_loss))
            logging.debug('test: time elapsed: %.3fs.'%(time.time()-start_time))
            for j in range(BATCH_SIZE):
                print(type(cur_batch[0][j]))
                #print(cur_batch[0][j].shape)
                img = cur_batch[0][j]/225
                
#                 img[:,:,0] = numpy.ones([5,5])*64/255.0
#                 img[:,:,1] = numpy.ones([5,5])*128/255.0
#                 img[:,:,2] = numpy.ones([5,5])*192/255.0
#                 print("===========data=============")
#                 print(i*BATCH_SIZE+j)
#                 print(train_search[i*BATCH_SIZE+j])
#                 #print(cur_batch[0][j])
                print(fc4[j])
                print(cur_batch[2][j])
#                 print("============data============")
#                 images = tf.image.decode_jpeg(cur_batch[0], channels=3)
# #                 Image.fromarray(np.asarray(image)).show()
#                 img = cv2.imread(train_search[i*BATCH_SIZE+j],1)
                height, width, channels = img.shape 
                gt1=(int(width/10*float(cur_batch[2][j][0])),int(height/10*float(cur_batch[2][j][1])))
                gt2=(int(width/10*float(cur_batch[2][j][2])),int(height/10*float(cur_batch[2][j][3])))
                cv2.rectangle(img, gt1, gt2, (0,255,0), 2, 1)
                fcp1=(int(width/10*float(fc4[j][0])),int(height/10*float(fc4[j][1])))
                fcp2=(int(width/10*float(fc4[j][2])),int(height/10*float(fc4[j][3])))
                cv2.rectangle(img, fcp1, fcp2, (255,0,0), 2, 1)
              
                print("load=================================================================")
                cv2.imshow("output", img)  
                # Show image
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("get keyboard interrupt")

