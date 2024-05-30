def set_template(args):
    # Set the templates here
    if args.template.find('mst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Phi'
        args.learning_rate = 1e-4
        args.batch_size = 6

    if args.template.find('gap_net') >= 0 or args.template.find('admm_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.milestones = range(30,args.max_epoch,30)
        args.gamma = 0.9
        args.learning_rate = 1e-3

    if args.template.find('tsa_net') >= 0:
        args.input_setting = 'HM'
        args.input_mask = None

    if args.template.find('hdnet') >= 0:
        args.input_setting = 'H'
        args.input_mask = None

    if args.template.find('dgsmp') >= 0:
        args.input_setting = 'Y'
        args.input_mask = None
        args.batch_size = 2
        args.milestones = [args.max_epoch]
        args.learning_rate = 1e-4

    if args.template.find('birnat') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        args.batch_size = 1
        args.max_epoch = 100
        args.milestones = [args.max_epoch]
        args.learning_rate = 1.5e-4

    if args.template.find('mst_plus_plus') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.learning_rate = 1e-4
    
    if args.template.find('cst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.learning_rate = 4e-4
        args.max_epoch = 500

    if args.template.find('dnu') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.batch_size = 2
        args.max_epoch = 150
        args.milestones = range(10,args.max_epoch,10)
        args.gamma = 0.9
        args.learning_rate = 4e-4

    if args.template.find('lambda_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        args.learning_rate = 1.5e-4

    if args.template.find('zxy') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('naf') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('scu') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        
    if args.template.find('step') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
    
    if args.template.find('nerf') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask_small'
        args.scheduler = 'CosineAnnealingLR'
        
    if args.template.find('nerf_nosr') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('dim') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('dun') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.learning_rate = 4e-4
        args.gpu_id = '0, 1'

    if args.template.find('rnn') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.learning_rate = 4e-4

    if args.template.find('e2e') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.batch_size = 4
        args.milestones = [args.max_epoch]
        args.learning_rate = 1e-4