def set_template(args):
    if args.template.find('pca') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.learning_rate = 4e-4
        args.gpu_id = '0, 1, 2, 3'
