
classdef configuration

    properties ( Constant = false )
        
        %% Experiment Name
        exp_name='dataset_1_9_channels_5_A';

        %GPU to use
        GPU_Number = -1; %negative number will use cpu

        %% Random seed value.
        seed=0;

        %% Random seed value. by default dataset 1 corressponds to IITM_Task_Independent_Subject_Regcognition_Dataset
        dataset = 1;
        

        %% Directory in which the continues data are stored; Data needs to be present for the program to run.
        data_dir_128='../IITM_Task_Independent_EEG_Dataset/';

        %% split to chosen
        split = 15
        splited_dir = '../splited_data/'

        %% split for validation
        val_per=0.10

        %% number of times the classification should be repeated.
        trials = 1;
        
        %% test only. if set to one training part will be skipped.
        test_only = 0;        
        
        %% Directory in which the models will be saved for training and testing will be stored; will be created by the program.
        base_history_dir='../model_history/';
        history_save = '';
        
        %% number of channels in EEG data
        num_channels = 128;
        
        %% Name of the feature to be used
        feature_name = {...
        'spectral_power',...            %1
        'LFCC',...                      %2
        'multitaper_spectrogram',...    %3
        'raw_time_series',...           %4
        };
        feature = 1;
        feature_function={@average_power_spectrogram, @LFCC_from_front_end_dsp, @multitaper_spectrum, @get_raw_time_series};
        
        
        %% Name of the classifier to be used
        classifier_name={...
        'modified-i-vector_Cosine',...      %1 %working
        'modified-x-vector_Cosine',...      %2 %working
        'ix-vector_cosine',...              %43 %working
        };
        classifier = 2;
        
        classifier_function={
            @classify_using_i_vector_chan_stat_cos,...
            @classify_using_mod_x_vector,...
            @classify_using_ix_vector_cosine,...
            };
        
        
        %% Name of the lobe to be used
        lobe_name = {...
        'All_Channels',...      %1
        'Frontal',...           %2
        'Parietal',...          %3
        'Temporal',....         %4
        'Occipital',...         %5
        '9-Channel',...         %6
        '16-Channel',...        %7
        '32-Channel',...        %8
        '64-Channel',...        %9
        'frontal',...           %10
        'central',...           %11
        'parietal',...          %12
        'occipital',...         %13
        'temporal',...          %14
        '6_channel',...         %15
        '4_channel',...         %16
        };
        
        lobe = 6;
        lobe_map = {...
            1:128,...
            [1 2 3 4 8 9 10 11 14 15 16 18 19 21 22 23 24 25 26 27 32 33 38 121 122 123 124],...
            [52 53 54 55 58 59 60 61 62 63 64 72 77 78 79 85 86 91 92 95 96 99 100 107],...
            [43 44 48 49 50 56 57 101 113 114 119 120],...
            [65 66 68 69 70 71 73 74 75 76 81 82 83 84 88 89 90 94],...
            [11, 70, 83, 36, 104, 122, 33, 58, 96],...
            [1:8:128],...
            [1:4:128],...
            [1:2:128],...
            [17,11,39,115,32,1,23,3,15],...
            [29,111,36,104,47,98,40,109,55],...
            [62,63,99,58,96,52,92,54,79],...
            [75,73,88,66,84,68,94,70,83],...
            [45,49,57,44,119,108,113,100,114],...
            [11,75,58,96,36,104],...
            [58,96,36,104],...                               `     
            };
        
        
        %% Sampling rate used to record data
        samp_rate =250;
        
        %% window size in ms
        win_size = 360;
        
        
        %% overlap size in ms
        overlap = 0;
        
        %% no of fft points
        nfft = 256
        
        %% low frequency limit in Hz
        %todo: replace this with bands
        lfreq = 3
        
        %% high frequency limit in Hz
        %todo: replace this with bands
        hfreq = 30
        
        %% Dir for saving the features
        features_base_dir='../features/';
        features_dir='';
        
        
        %% FFT order (log_{2} of fft size) --> need only for LFCC
        fftorder = 10;
        
        %% Number of filters --> need only for LFCC
        numfilters = 10;
        
        %% Number of Ceps --> need only for LFCC
        numceps = 4;
        
        %% Delta Needed ?--> need only for LFCC
        delta_1 = 1;
        
        %% Delta^2 needed? --> needed only for LFCC
        delta_2 = 1;
        
        %% hidden layer config for x-vector
        hiddenlayers = [1024 512 160];
        
        %% UBM Parameters
        mixtures=7;
        ivec_dim=100;
        initsize=50;
        top_c=10;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters that
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% doesn't change
        
        %% Directory to save models
        ubm_model_dir = 'ubm_model';
        ivec_prog_mat_dir = 'ivec_prog_mat_files';
        dnn_model_files = 'dnn_models';
        
        %% tmp dir to save ubm data
        per_tmp_dir = '../tmp';
        per_ubm_data_dir = 'ubm_data';
        per_ubm_temp_dir = 'ubm_tmp_files';
        per_ubm_train_data_dir = 'ubm_train';
        per_ubm_val_data_dir = 'ubm_val';
        per_ubm_test_data_dir = 'ubm_test';
        per_ivec_temp_dir = 'ivec_tmp_files';


        data_dir= '';

        tmp_dir = '';
        ubm_data_dir = '';
        ubm_temp_dir = '';
        ubm_train_data_dir = '';
        ubm_val_data_dir = '';
        ubm_test_data_dir = '';
        ivec_temp_dir = '';
        
        
    end

    methods 
        function self = configuration(self)
            %% Directory in which the models will be saved for training and testing will be stored; will be created by the program.
            self.history_save=[self.base_history_dir,'/',self.exp_name];

            %% Dir for saving the features
            self.features_dir=[self.features_base_dir,self.exp_name,];


            self.data_dir = self.data_dir_128;
            
            %% UpdateTMPDir
            self.tmp_dir=[self.per_tmp_dir,'/',self.exp_name,'/'];
            self.ubm_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_data_dir];
            self.ubm_temp_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_temp_dir];
            self.ubm_train_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_train_data_dir];
            self.ubm_val_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_val_data_dir];
            self.ubm_test_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_test_data_dir];
            self.ivec_temp_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ivec_temp_dir];
        end

        function self = updateModelLocation(self)
            %% Directory in which the models will be saved for training and testing will be stored; will be created by the program.
            self.history_save=['../model_history/','/',self.exp_name];
            
        end

        function self = updateFeaturesLocation(self, exp_name)
            %% Directory in which the models will be saved for training and testing will be stored; will be created by the program.
            self.features_dir=[self.features_base_dir, exp_name];

            
        end

        function self = updateTmpLocation(self)
             %% UpdateTMPDir
            self.tmp_dir=[self.per_tmp_dir,'/',self.exp_name,'/'];
            self.ubm_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_data_dir];
            self.ubm_temp_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_temp_dir];
            self.ubm_train_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_train_data_dir];
            self.ubm_val_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_val_data_dir];
            self.ubm_test_data_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ubm_test_data_dir];
            self.ivec_temp_dir=[self.per_tmp_dir,'/',self.exp_name,'/',self.per_ivec_temp_dir];
        end

        function self = updateCustomTmpLocation(self,tmp_name)
             %% UpdateTMPDir
            self.tmp_dir=[self.per_tmp_dir,'/',tmp_name,'/'];
            self.ubm_data_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ubm_data_dir];
            self.ubm_temp_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ubm_temp_dir];
            self.ubm_train_data_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ubm_train_data_dir];
            self.ubm_val_data_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ubm_val_data_dir];
            self.ubm_test_data_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ubm_test_data_dir];
            self.ivec_temp_dir=[self.per_tmp_dir,'/',tmp_name,'/',self.per_ivec_temp_dir];
        end
    end
end





