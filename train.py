#TODO Setup parameters in config files
#TODO Setup ReadME

from pykeen.pipeline import pipeline
import dotenv

dotenv.load_dotenv(override=True)

pipeline_result = pipeline(
     dataset='wn18rr',
     model='TuckER',
     loss='bcewithlogits',
     training_loop='sLCWA', # maybe needs to be changed
     negative_sampler='basic', # maybe needs to be changed
     training_kwargs=dict(
       num_epochs=500,
       batch_size = 128,
       checkpoint_name='my_checkpoint.pt',
       checkpoint_frequency=50,
       checkpoint_directory='checkpoints',
       label_smoothing = 0.1
    ),
     model_kwargs=dict(
        embedding_dim = 200 ,
        relation_dim = 30,
        dropout_0 = 0.2,
        dropout_1 = 0.2,
        dropout_2 = 0.3,
     ),
     optimizer = 'Adam',
     optimizer_kwargs = dict(
         lr = 0.003,
     ),
     lr_scheduler='ExponentialLR',
     lr_scheduler_kwargs=dict(
         gamma=1.0,
     ),
     result_tracker='wandb',
     result_tracker_kwargs=dict(
        project='tucker_wn18rr',
        entity = 'eth_ai_center_kg_project'
     ),
)
pipeline_result.save_to_directory('results/wn18rr_tucker')
