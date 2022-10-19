def yaml_to_rt_pipeline(config,filename='out'):

    out="[Version]\nAppVersion=5.3\nVersion=327\n"

    transformation_dict=config.__dict__

    for transformation in transformation_dict.keys():
        out+=f'\n[{transformation}]\n'
        hyperparameters_dict=transformation_dict[transformation]
        for parameter in hyperparameters_dict.keys():
            out+=f'{parameter}={hyperparameters_dict[parameter]}\n'

    out=out.replace('_',' ')

    with open(f'Pipelines/{filename}.pp3', 'w') as file:
        file.write(out)

    
