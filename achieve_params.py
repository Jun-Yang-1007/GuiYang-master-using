import tensorflow as tf
Flags = tf.flags.FLAGS
command_params = tf.app.flags.FLAGS
# 数据库固定参数
tf.app.flags.DEFINE_string('hostname', '', 'hostname')
tf.app.flags.DEFINE_string('port', '', 'port')
tf.app.flags.DEFINE_string('username', '', 'username')
tf.app.flags.DEFINE_string('password', '', 'password')
tf.app.flags.DEFINE_string('database', '', 'database')
tf.app.flags.DEFINE_string('table_name', '', 'table_name')
tf.app.flags.DEFINE_string('db_type', '', 'db_type')
tf.app.flags.DEFINE_string('activity_log_id', '', 'activity_log_id')
tf.app.flags.DEFINE_string('file_id', '', 'file_id')

# 模型参数
tf.flags.DEFINE_integer('unit', '30', 'number of units')
tf.flags.DEFINE_integer('batch_size', '32', 'batch size')
tf.flags.DEFINE_float('validation_split', '0.1', 'validation split')
tf.flags.DEFINE_integer('epoch', '20', 'iteration epochs')
tf.flags.DEFINE_string('type', 'easy', 'type of model')
tf.flags.DEFINE_float('dropout', '0.5', 'dropout')
tf.flags.DEFINE_integer('predict_length', '72', 'predict_length')
tf.flags.DEFINE_integer('look_back', '500', 'look_back')