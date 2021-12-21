class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            root_dir = r'E:\DataSet\DataSet\Object_detection\UCF-101'
            output_dir = r'E:\DataSet\train_val_test\object_recognition\ucf101'
            return root_dir, output_dir
        elif database == '':
            root_dir = ''
            output_dir = r'E:\DataSet\train_val_test\object_recognition\hmdb51'
            return root_dir, output_dir
        else:
            print('Database {} not available'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return r'E:\DataSet\pretrained\object_recognition\ucf101\ucf101-caffe.pth'
