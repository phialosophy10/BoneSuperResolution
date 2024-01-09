import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, LeakyReLU, Concatenate, Add, Lambda, BatchNormalization, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tqdm import tqdm as tqdm
from numba import jit
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata 
from tensorflow.keras.models import model_from_json
import os
xyz=np.load(file="./channel_flow_2dyz_48x48.npy")
z=np.reshape(xyz[:,0],(48,48))
y=np.reshape(xyz[:,1],(48,48))
z=z[0,:]
y=y[:,0]
y
xx=[]
yy=[]
zz=[]
xnew = np.linspace(0,6.25,64)
ynew = y
znew = np.linspace(0,3.2,48)
for i in range(48):
    for j in range(48):
        for kk in range(64):
                yy.append(ynew[i])
                zz.append(znew[j])
                xx.append(xnew[kk])


yy_np_0=np.array(yy)
zz_np_0=np.array(zz)
xx_np_0=np.array(xx)
yyy=yy_np_0.reshape((48,48,64))
zzz=zz_np_0.reshape((48,48,64))
xxx=xx_np_0.reshape((48,48,64))
yyy_tf=tf.convert_to_tensor(yyy,dtype=tf.float32)
zzz_tf=tf.convert_to_tensor(zzz,dtype=tf.float32)
xxx_tf=tf.convert_to_tensor(xxx,dtype=tf.float32)
xxx_tf_delt=tf.subtract(xxx_tf[:,:,1:],xxx_tf[:,:,:-1])
xxx_tf_delt
yyy_tf_delt=tf.subtract(yyy_tf[1:,:,:],yyy_tf[:-1,:,:])
zzz_tf_delt=tf.subtract(zzz_tf[:,1:,:],zzz_tf[:,:-1,:])
zzz_tf_delt




def CFE1():
    vgg = model_from_json(open('E:/linqi_doc/feature_extractor_cf/autoencoder_architecture.json').read()) 
    vgg.load_weights('./autoencoder_weights.hdf5')
    
    block3_conv3_copy = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same',name='out_put_before_cactivation')
    
    
    injection_model = Sequential(vgg.layers[:2] + [block3_conv3_copy])
    block3_conv3_copy.set_weights(vgg.layers[2].get_weights())
    img = Input(shape=(48,48,64,3))
    # Extract image features
    img_features = injection_model(img)
    # Create model and compile
    model = Model(img, img_features)
    model.trainable = False
    return model
 

def CFE2():
    vgg = model_from_json(open('./autoencoder_architecture.json').read()) 
    vgg.load_weights('./autoencoder_weights.hdf5')
    block3_conv3_copy = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',name='out_put_before_cactivation')
    injection_model = Sequential(vgg.layers[:5] + [block3_conv3_copy])
    block3_conv3_copy.set_weights(vgg.layers[5].get_weights())
    img = Input(shape=(48,48,64,3))
    # Extract image features
    img_features = injection_model(img)
    # Create model and compile
    model = Model(img, img_features)
    model.trainable = False
    return model


def build_generator():
        def generator_loss(y_true,y_pred): 
            #percept_loss
            gen_image = y_pred
            ori_image = y_true
            
            #adv_loss
            discriminator.trainable = False
            fake_d_out,real_d_out=discriminator([gen_image, ori_image])           
            fake_logit= tf.sigmoid((fake_d_out - K.mean(real_d_out)))
            adv_fake_logit = tf.keras.losses.binary_crossentropy(K.ones_like(fake_logit), fake_logit)
            adv_loss=adv_fake_logit          
            #pixel_loss
            pixel_loss_o = tf.losses.mean_squared_error(gen_image, ori_image)
            pixel_loss=K.mean(pixel_loss_o)
            gen_f1=CFE1(gen_image)
            gen_f2=CFE2(gen_image)
            ori_f1=CFE1(ori_image)
            ori_f2=CFE2(ori_image)
            percept_loss_mse1 = tf.losses.mean_squared_error(gen_f1,ori_f1)
            percept_loss1 = K.mean(percept_loss_mse1)                        
            percept_loss_mse2 = tf.losses.mean_squared_error(gen_f2,ori_f2)
            percept_loss2 = K.mean(percept_loss_mse2)
            percept_loss = 0.2*percept_loss1 + 0.8*percept_loss2 
            grad_u_mse=tf.zeros((1))
            grad_v_mse=tf.zeros((1))
            grad_w_mse=tf.zeros((1))
            for i in range(8):
                    upred=y_pred[i,:,:,:,0]
                    utrue=y_true[i,:,:,:,0]
                    d_upred=tf.subtract(upred[:,:,1:],upred[:,:,:-1])
                    d_utrue=tf.subtract(utrue[:,:,1:],utrue[:,:,:-1])
                    grad_u_pred=d_upred/xxx_tf_delt
                    grad_u_ture=d_utrue/xxx_tf_delt
                    grad_u_mse += tf.losses.mean_squared_error(grad_u_pred,grad_u_ture)
                    vpred=y_pred[i,:,:,:,1]
                    vtrue=y_true[i,:,:,:,1]
                    d_vpred=tf.subtract(vpred[1:,:,:],vpred[:-1,:,:])
                    d_vtrue=tf.subtract(vtrue[1:,:,:],vtrue[:-1,:,:])
                    grad_v_pred=d_vpred/yyy_tf_delt
                    grad_v_ture=d_vtrue/yyy_tf_delt 
                    grad_v_mse += tf.losses.mean_squared_error(grad_v_pred,grad_v_ture)
                    wpred=y_pred[i,:,:,:,2]
                    wtrue=y_true[i,:,:,:,2]
                    d_wpred=tf.subtract(wpred[:,1:,:],wpred[:,:-1,:])
                    d_wtrue=tf.subtract(wtrue[:,1:,:],wtrue[:,:-1,:])
                    grad_w_pred=d_wpred/zzz_tf_delt
                    grad_w_ture=d_wtrue/zzz_tf_delt
                    grad_w_mse += tf.losses.mean_squared_error(grad_w_pred,grad_w_ture)
            

            
            loss_grad=(grad_u_mse+grad_v_mse+grad_w_mse)/16/3
            
               
            
            # momentum loss
            loss_m_1_mse=tf.zeros((1))
            loss_m_2_mse=tf.zeros((1))
            loss_m_3_mse=tf.zeros((1))
            
            for i in range(7):
                upred0=y_pred[i,:,:,:,0]
                utrue0=y_true[i,:,:,:,0]
                
                upred1=y_pred[i+1,:,:,:,0]
                utrue1=y_true[i+1,:,:,:,0]
                
                vpred0=y_pred[i,:,:,:,1]
                vtrue0=y_true[i,:,:,:,1]
                vpred1=y_pred[i+1,:,:,:,1]
                vtrue1=y_true[i+1,:,:,:,1]
                
                
                wpred0=y_pred[i,:,:,:,2]
                wtrue0=y_true[i,:,:,:,2]
                wpred1=y_pred[i+1,:,:,:,2]
                wtrue1=y_true[i+1,:,:,:,2]
                

                du_dt_p=tf.subtract(upred1,upred0)/0.0078
                du_dt_t=tf.subtract(utrue1,utrue0)/0.0078
                
                
                du_dt_p_cut=du_dt_p[1:,1:,1:]
                du_dt_t_cut=du_dt_t[1:,1:,1:]
  
                d_upred_x=tf.subtract(upred1[:,:,1:],upred1[:,:,:-1])
                d_utrue_x=tf.subtract(utrue1[:,:,1:],utrue1[:,:,:-1])
                
                grad_u_pred_x=d_upred_x/xxx_tf_delt
                grad_u_ture_x=d_utrue_x/xxx_tf_delt
                
                grad_u_x_pred_u=grad_u_pred_x*upred1[:,:,1:]
                grad_u_x_ture_u=grad_u_ture_x*utrue1[:,:,1:]
                
                grad_u_x_pred_u_cut=grad_u_x_pred_u[1:,1:,:]
                grad_u_x_ture_u_cut=grad_u_x_ture_u[1:,1:,:]
                
       
                d_upred_y=tf.subtract(upred1[1:,:,:],upred1[:-1,:,:])
                d_utrue_y=tf.subtract(utrue1[1:,:,:],utrue1[:-1,:,:])
                
                grad_u_pred_y=d_upred_y/yyy_tf_delt
                grad_u_ture_y=d_utrue_y/yyy_tf_delt
                
                grad_u_pred_y_v=grad_u_pred_y*vpred1[1:,:,:]
                grad_u_ture_y_v=grad_u_ture_y*vtrue1[1:,:,:]
                
                grad_u_pred_y_v_cut=grad_u_pred_y_v[:,1:,1:]
                grad_u_ture_y_v_cut=grad_u_ture_y_v[:,1:,1:]
                
     
                d_upred_z=tf.subtract(upred1[:,1:,:],upred1[:,:-1,:])
                d_utrue_z=tf.subtract(utrue1[:,1:,:],utrue1[:,:-1,:])
                
                grad_u_pred_z=d_upred_z/zzz_tf_delt
                grad_u_ture_z=d_utrue_z/zzz_tf_delt
                
                grad_u_pred_z_w=grad_u_pred_z*wpred1[:,1:,:]
                grad_u_ture_z_w=grad_u_ture_z*wtrue1[:,1:,:]
                
                grad_u_pred_z_w_cut=grad_u_pred_z_w[1:,:,1:]
                grad_u_ture_z_w_cut=grad_u_ture_z_w[1:,:,1:]
                
                
                loss_m_1_p = du_dt_p_cut + grad_u_x_pred_u_cut + grad_u_pred_y_v_cut + grad_u_pred_z_w_cut
                loss_m_1_t = du_dt_t_cut + grad_u_x_ture_u_cut + grad_u_ture_y_v_cut + grad_u_ture_z_w_cut
                
                loss_m_1_mse += tf.losses.mean_squared_error(loss_m_1_p,loss_m_1_t)
                
           
                dv_dt_p=tf.subtract(vpred1,vpred0)/0.0078
                dv_dt_t=tf.subtract(vtrue1,vtrue0)/0.0078
                
                
                dv_dt_p_cut=dv_dt_p[1:,1:,1:]
                dv_dt_t_cut=dv_dt_t[1:,1:,1:]
                
           
                
                d_vpred_x=tf.subtract(vpred1[:,:,1:],vpred1[:,:,:-1])
                d_vtrue_x=tf.subtract(vtrue1[:,:,1:],vtrue1[:,:,:-1])
                
                grad_v_pred_x=d_vpred_x/xxx_tf_delt
                grad_v_ture_x=d_vtrue_x/xxx_tf_delt
                
                grad_v_x_pred_u=grad_v_pred_x*upred1[:,:,1:]
                grad_v_x_ture_u=grad_v_ture_x*utrue1[:,:,1:]
                
                grad_v_x_pred_u_cut=grad_v_x_pred_u[1:,1:,:]
                grad_v_x_ture_u_cut=grad_v_x_ture_u[1:,1:,:]
                
            
                d_vpred_y=tf.subtract(vpred1[1:,:,:],vpred1[:-1,:,:])
                d_vtrue_y=tf.subtract(vtrue1[1:,:,:],vtrue1[:-1,:,:])
                
                grad_v_pred_y=d_vpred_y/yyy_tf_delt
                grad_v_ture_y=d_vtrue_y/yyy_tf_delt
                
                grad_v_pred_y_v=grad_v_pred_y*vpred1[1:,:,:]
                grad_v_ture_y_v=grad_v_ture_y*vtrue1[1:,:,:]
                
                grad_v_pred_y_v_cut=grad_v_pred_y_v[:,1:,1:]
                grad_v_ture_y_v_cut=grad_v_ture_y_v[:,1:,1:]
                
             
                d_vpred_z=tf.subtract(vpred1[:,1:,:],vpred1[:,:-1,:])
                d_vtrue_z=tf.subtract(vtrue1[:,1:,:],vtrue1[:,:-1,:])
                
                grad_v_pred_z=d_vpred_z/zzz_tf_delt
                grad_v_ture_z=d_vtrue_z/zzz_tf_delt
                
                grad_v_pred_z_w=grad_v_pred_z*wpred1[:,1:,:]
                grad_v_ture_z_w=grad_v_ture_z*wtrue1[:,1:,:]
                
                grad_v_pred_z_w_cut=grad_v_pred_z_w[1:,:,1:]
                grad_v_ture_z_w_cut=grad_v_ture_z_w[1:,:,1:]
                
                
                loss_m_2_p = dv_dt_p_cut + grad_v_x_pred_u_cut + grad_v_pred_y_v_cut + grad_v_pred_z_w_cut
                loss_m_2_t = dv_dt_t_cut + grad_v_x_ture_u_cut + grad_v_ture_y_v_cut + grad_v_ture_z_w_cut
                
                
                loss_m_2_mse += tf.losses.mean_squared_error(loss_m_2_p,loss_m_2_t)
                
                dw_dt_p=tf.subtract(wpred1,wpred0)/0.0078
                dw_dt_t=tf.subtract(wtrue1,wtrue0)/0.0078
                
                
                dw_dt_p_cut=dw_dt_p[1:,1:,1:]
                dw_dt_t_cut=dw_dt_t[1:,1:,1:]
                
                
                
                
                
                d_wpred_x=tf.subtract(wpred1[:,:,1:],wpred1[:,:,:-1])
                d_wtrue_x=tf.subtract(wtrue1[:,:,1:],wtrue1[:,:,:-1])
                
                grad_w_pred_x=d_wpred_x/xxx_tf_delt
                grad_w_ture_x=d_wtrue_x/xxx_tf_delt
                
                grad_w_x_pred_u=grad_w_pred_x*upred1[:,:,1:]
                grad_w_x_ture_u=grad_w_ture_x*utrue1[:,:,1:]
                
                grad_w_x_pred_u_cut=grad_w_x_pred_u[1:,1:,:]
                grad_w_x_ture_u_cut=grad_w_x_ture_u[1:,1:,:]
                
                
                
                
                
                
                d_wpred_y=tf.subtract(wpred1[1:,:,:],wpred1[:-1,:,:])
                d_wtrue_y=tf.subtract(wtrue1[1:,:,:],wtrue1[:-1,:,:])
                
                grad_w_pred_y=d_wpred_y/yyy_tf_delt
                grad_w_ture_y=d_wtrue_y/yyy_tf_delt
                
                grad_w_pred_y_v=grad_w_pred_y*vpred1[1:,:,:]
                grad_w_ture_y_v=grad_w_ture_y*vtrue1[1:,:,:]
                
                grad_w_pred_y_v_cut=grad_w_pred_y_v[:,1:,1:]
                grad_w_ture_y_v_cut=grad_w_ture_y_v[:,1:,1:]
                
                
                
                
                
                
                d_wpred_z=tf.subtract(wpred1[:,1:,:],wpred1[:,:-1,:])
                d_wtrue_z=tf.subtract(wtrue1[:,1:,:],wtrue1[:,:-1,:])
                
                grad_w_pred_z=d_wpred_z/zzz_tf_delt
                grad_w_ture_z=d_wtrue_z/zzz_tf_delt
                
                grad_w_pred_z_w=grad_w_pred_z*wpred1[:,1:,:]
                grad_w_ture_z_w=grad_w_ture_z*wtrue1[:,1:,:]
                
                grad_w_pred_z_w_cut=grad_w_pred_z_w[1:,:,1:]
                grad_w_ture_z_w_cut=grad_w_ture_z_w[1:,:,1:]
                
                
                loss_m_3_p = dw_dt_p_cut + grad_w_x_pred_u_cut + grad_w_pred_y_v_cut + grad_w_pred_z_w_cut
                loss_m_3_t = dw_dt_t_cut + grad_w_x_ture_u_cut + grad_w_ture_y_v_cut + grad_w_ture_z_w_cut
                
                
                
                loss_m_3_mse += tf.losses.mean_squared_error(loss_m_3_p,loss_m_3_t)
                
          
            loss_momentum = (loss_m_1_mse + loss_m_2_mse + loss_m_3_mse)/7/3

            #total loss
            G_loss= cofn1*pixel_loss + cofn2*adv_loss+cofn3*percept_loss +cofn4* loss_grad + cofn5*loss_momentum

    
            return G_loss
        
        #network
        def dense_block(input):
            c=32
            x1 = Conv3D(c, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])
            x2 = Conv3D(c, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([x1,  x2])
            x3 = Conv3D(c, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([x2, x3])
            x4 = Conv3D(c, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([x3, x4])  
            x5 = Conv3D(c, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x) 
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample_1(x,names):
            x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
            x = UpSampling3D((2,2,2))(x) 
            return x
        
        def upsample_2(x,names):
            x = Conv3D(64, kernel_size=5, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=5, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=5, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = UpSampling3D((2,2,2))(x)          
            return x
        
        def upsample_3(x,names):
            x = Conv3D(64, kernel_size=9, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=9, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv3D(64, kernel_size=9, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = UpSampling3D((2,2,2))(x)
            return x

        # Input low resolution image
        
        data_input = Input((12,12,16,3))
        
        x_start = Conv3D(32, kernel_size=3, strides=1, padding='same')(data_input)
        
        # Residual-in-Residual Dense Block(23个)
        x = RRDB(x_start)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x) 
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        x = RRDB(x)
        
        
        # Post-residual block
        x = Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
        # x = Lambda(lambda x: x * 0.2)(x)
        x_b = Add()([x, x_start])
        x_b1=x_b
        x_b2=x_b
        x_b3=x_b
        # Upsampling depending on factor
        for i_1 in range(upscaling_times):
            x_b1 = upsample_1(x_b1,i_1)
        for i_2 in range(upscaling_times):
            x_b2 = upsample_2(x_b2,i_2)  
        for i_3 in range(upscaling_times):
            x_b3 = upsample_3(x_b3,i_3)
        x_u=Add()([x_b1,x_b2,x_b3])

        x = Conv3D(32, kernel_size=3, strides=1, padding='same')(x_u)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv3D(3, kernel_size=3, strides=1, padding='same', activation='sigmoid',name='conv_output')(x)
        
        # Create model and compile
        model = Model(inputs=data_input, outputs=hr_output)
        

        model.compile(optimizer=Adam(gen_lr, 0.9,0.999),metrics=['accuracy',PSNR], loss= generator_loss) 
        return model
    
def build_discriminator(): 
        def discriminator_network(filters=32):
            def conv3d_block(input, filters, strides):
                d = Conv3D(filters, kernel_size=3, strides=strides, padding='same')(input)
                d = BatchNormalization(momentum=0.8)(d)
                d = LeakyReLU(alpha=0.2)(d)
                return d
    
            # Input high resolution image
            img = Input(shape=(48,48,64,3))
    
            x = Conv3D(filters, kernel_size=3, strides=1, padding='same')(img)
            x = LeakyReLU(alpha=0.2)(x)
            
            x = conv3d_block(x, filters, strides=2)
            x = conv3d_block(x, filters * 2, strides=1)
            x = conv3d_block(x, filters * 2, strides=2)
            x = conv3d_block(x, filters * 4, strides=1)
            x = conv3d_block(x, filters * 4, strides=2)
            x = conv3d_block(x, filters * 8, strides=1)
            x = conv3d_block(x, filters * 8, strides=2)
            
            x= Conv3D(filters, kernel_size=3, strides=1, padding='same')(x)
            

            model = Model(inputs=img, outputs=x)
            return model   
    

        fake_hr_img=Input(shape=(48,48,64,3))
        real_hr_img=Input(shape=(48,48,64,3))
        
        discriminator_net = discriminator_network()
        fake_d_out = discriminator_net(fake_hr_img)
        real_d_out = discriminator_net(real_hr_img)
        
        model = Model([fake_hr_img,real_hr_img], [fake_d_out,real_d_out])
        

        fake_logit= tf.sigmoid((fake_d_out - K.mean(real_d_out)))
        real_logit= tf.sigmoid((real_d_out - K.mean(fake_d_out)))
        fake_logit_F = tf.keras.losses.binary_crossentropy(K.zeros_like(fake_logit), fake_logit)
        real_logit_F = tf.keras.losses.binary_crossentropy(K.ones_like(real_logit), real_logit)
        dis_loss0 = (K.mean(fake_logit_F)+K.mean(real_logit_F))/2 
        
        model.add_loss(dis_loss0)  
        model.metrics_tensors = []
        model.metrics_names.append('dis_loss')
        model.metrics_tensors.append(dis_loss0)

        def discriminator_loss(y_true,y_pred):  

            dis_loss = dis_loss0
            return dis_loss 
        
        #编译
        model.compile(optimizer=Adam(dis_lr,0.9,0.999), loss= discriminator_loss)  
        return model
    
def PSNR( y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)    
  
@jit
def train_srgan(epochs=20000, batch_size=10,sample_interval=50):
        
        g_loss_store=[]
        d_loss_store=[]
        g_acc=[]
        g_psnr=[]
        start = 0
        ac=0

        for epoch in tqdm(range(epochs)):
            stop = start + batch_size
            lr_data_input=lr_data[start: stop]

            hr_3d_input=hr_data[start: stop]

            generated_hrs = generator.predict(lr_data_input)

            #Train discriminator
            discriminator_loss = discriminator.train_on_batch([generated_hrs,hr_3d_input],None)
           
            d_loss_store.append(discriminator_loss[0])

            #Train generator
            generator_loss = generator.train_on_batch(lr_data_input, hr_3d_input)
            g_loss_store.append(generator_loss[0])
            g_acc.append(generator_loss[1])
            g_psnr.append(generator_loss[2])
            

            if epoch%20000==0 and epoch!=0:
                print('\n') 
                print('changed learn rate.')
                lr = K.get_value(generator.optimizer.lr) 
                lr = lr * 0.5 
                print('learning rate is:'+str(lr))
                K.set_value(generator.optimizer.lr, lr) 
            
            start += batch_size
            if start > len(lr_data)-batch_size:
                start = 0
            if epoch % 50 == 0:
                print("\n")
                print('discriminator loss:', discriminator_loss[0])
                print('generator loss:', generator_loss)     
            if epoch % sample_interval == 0:
                print("outputing...")
                
                
                
                file_name="./loss/"+name_save+"loss_file.csv"
                save_data = pd.DataFrame({'g_loss':g_loss_store, 'd_loss':d_loss_store,'g_acc':g_acc,'g_psnr':g_psnr})
                save_data.to_csv(file_name) 
                #save architecture
                if ac==0:
                    ac=ac+1
                    json_string =generator.to_json()  
                    open('./architecture/'+name_save+'_generator_architecture.json','w').write(json_string)
                #save weights
                if epoch > 50000 and epoch % 1000 == 0:
                    generator.save_weights('./weights/'+name_save+'_generator_weights'+'{0:01d}'.format(epoch)+'.h5')
                if epoch % 100 == 0:
                    Data1=generated_hrs
                    Data=Data1[:,:,23:24,:,:]
                    Asix=np.load(file="./channel_flow_2dxy_48x64.npy")
    
    
                    channel=0
                    
                    s_begin=0
                    s_end=1
                    step=1
                    save_name='ghr_top'
                    
                    figsize=5,2
                    
                    vmin_u=0
                    vmax_u=1
                    
                    
                     
                    
                    for i in tqdm(range(s_begin,s_end,step)):
                        Asix_y=Asix[:,1:2].reshape((48*64)).tolist()
                        Asix_x=Asix[:,0:1].reshape((48*64)).tolist()
                        velocity_u=Data[i:i+1,:,:,:,channel:channel+1].reshape((48*64)).tolist()
                        xi=np.linspace(min(Asix_x),max(Asix_x),500)
                        yi=np.linspace(min(Asix_y),max(Asix_y),500)
                        [X,Y]=np.meshgrid(xi,yi)
                        Velocity=griddata((Asix_x,Asix_y),velocity_u,(X,Y),method='linear')
                        figure= plt.subplots(figsize=figsize)
                        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
                        if channel==0:
                            c_name='u'
                            vmin=vmin_u
                            vmax=vmax_u
                        
                        im=plt.contourf(X,Y,Velocity,25,alpha=1,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
                        save_name_contour='./img/'+save_name+'_'+c_name+'_'+str(epoch)+"_"+str(i)+'.png'
                        plt.savefig(save_name_contour)
                
                    Data1=generated_hrs
                    Data=Data1[:,:,:,31:32,:]
                    Asix=np.load(file="./channel_flow_2dyz_48x48.npy")
    
    
                    channel=1
                    
                    s_begin=0
                    s_end=1
                    step=1
                    save_name='ghr_side'
                    
                    figsize=5,2
                    
                    vmin_u=0
                    vmax_u=1 
                      
                    
                     
                    
                    for i in tqdm(range(s_begin,s_end,step)):
                        Asix_y=Asix[:,1:2].reshape((48*48)).tolist() 
                        Asix_x=Asix[:,0:1].reshape((48*48)).tolist()
                        velocity_u=Data[i:i+1,:,:,:,channel:channel+1].reshape((48*48)).tolist()
                        xi=np.linspace(min(Asix_x),max(Asix_x),500)
                        yi=np.linspace(min(Asix_y),max(Asix_y),500)
                        [X,Y]=np.meshgrid(xi,yi)
                        Velocity=griddata((Asix_x,Asix_y),velocity_u,(X,Y),method='linear')
                        figure= plt.subplots(figsize=figsize)
                        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
                        if channel==0:
                            c_name='u'
                            vmin=vmin_u
                            vmax=vmax_u
                        
                        im=plt.contourf(X,Y,Velocity,25,alpha=1,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
                        save_name_contour='./img2/'+save_name+'_'+c_name+'_'+str(epoch)+"_"+str(i)+'.png'
                        plt.savefig(save_name_contour)
                
                
                
                
                

   
name_save="low_resolution"
# Low-resolution image dimensions

length_lr = 12
height_lr = 12
width_lr = 16




# High-resolution image dimensions
upscaling_factor = 4 
upscaling_times=2


# # Low-resolution and high-resolution shapes


# Load hr and lr data
lr_data=np.load(file="")
print(lr_data.shape)



hr_data=np.load(file="")
print(hr_data.shape)

# Learning rates 
gen_lr = 1e-4
dis_lr = 1e-4
 
# Gnerator loss cofficient
cofn1=1000 #pi 1000
cofn2=10   #a 10
cofn3=2000 #pe 2000
cofn4=1
cofn5=1


# Build the discriminator network
discriminator=build_discriminator()
# Build & compile the generator network
generator = build_generator() 



CFE1=CFE1()
CFE2=CFE2()
path_file = "./loss"
if not os.path.exists(path_file):
    os.mkdir(path_file)
path_file = "./architecture" 
if not os.path.exists(path_file):
    os.mkdir(path_file)
path_file = "./weights"
if not os.path.exists(path_file):
    os.mkdir(path_file)
    
path_file = "./img" 
if not os.path.exists(path_file):
    os.mkdir(path_file)
    
path_file = "./img2" 
if not os.path.exists(path_file):
    os.mkdir(path_file)    
    
    
if __name__ == '__main__':
    discriminator.summary()
    generator.summary()
    train_srgan(epochs=100000, batch_size=8, sample_interval=100)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
