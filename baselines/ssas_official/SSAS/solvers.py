from dataloader import *
import lr_schedule
import utils
from modules import z_score, normalize
import numpy as np
from utils import  LabelSmooth, discrepancy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import Adver_network
from new_network import MLPBase, feat_bottleneck, feat_classifier
import copy
try:
    from feature_mixstyle import FeatureMixStyle
except ImportError:
    FeatureMixStyle = None
try:
    from rspm_loss import ReliabilitySourcePrototypeMemory
except ImportError:
    ReliabilitySourcePrototypeMemory = None


def test_suda(loader, model):
    start_test = True
    with torch.no_grad():
        # get iterate data
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            # get sample and label
            # data = iter_test.next()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # load in gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # obtain predictions
            _, outputs = model(inputs)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # obtain labels
    _, predictions = torch.max(all_output, 1)
    # calculate accuracy for all examples
    accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

    y_true = all_label.cpu().data.numpy()
    y_pred = predictions.cpu().data.numpy()
    labels = np.unique(y_true)

    # Binarize ytest with shape (n_samples, n_classes)
    ytest = label_binarize(y_true, classes=labels)
    ypreds = label_binarize(y_pred, classes=labels)

    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, auc, matrix


def test_muda(dataset_test, netA,netB,netC,args):
    start_test = True
    features = None
    new_shape = (200, 62, 9 * 5)
    with torch.no_grad():

        for batch_idx, data in enumerate(dataset_test):
            Tx = data['Tx']
            Ty = data['Ty']
            Tx = Tx.float().cuda()
            # tmp_Tx = Tx.reshape(*new_shape)
            # tmp_x = augment(tmp_Tx).cuda()
            # obtain predictions
            # feats, outputs = model(Tx)
            feats = netB(netA(Tx))
            outputs = netC(feats)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = Ty.float()
                features = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

            # obtain labels
        _, predictions = torch.max(all_output, 1)
        # calculate accuracy for all examples
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)

        # Binarize ytest with shape (n_samples, n_classes)
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)

        return accuracy, f1, auc, matrix, features, y_pred

def NEW_DGMA(X, Y, Domain_label, count_num, netF, args):
    """
    Parameters:
        @args: arguments
    """
    # lmmd_loss_instance = lmmd.LMMD_loss()  # 创建 LMMD_loss 类的实例
    # --------------------------
    # Prepare data
    # --------------------------
    # select target subject
    trg_subj = args.target - 1
    count_domain = 0
    for i in range(len(X)):
        #如果有权重为0.1的话，就将这个数据剔除
        # if count_num[i] == 0.1:
        #    count_domain += 1
        # else:
        #    X[i] = count_num[i] * X[i] 
        X[i] = count_num[i] * X[i] 
    # Target data
    Tx = np.array(X[trg_subj])
    Ty = np.array(Y[trg_subj])
    subject_ids = X.keys()
    num_domains = len(subject_ids)
    Vx = Tx
    Vy = Ty

    # Standardize target data
    Tx, m, std = z_score(Tx)
    Vx = normalize(Vx, mean=m, std=std)

    print("Target subject:", trg_subj)
    print("Tx:", Tx.shape, " Ty:", Ty.shape)
    print("Vx:", Vx.shape, " Vy:", Vy.shape)
    print("Num. domains:", num_domains)
    print("Data were succesfully loaded")

    train_loader = UnalignedDataLoader_domain()
    train_loader.initialize(num_domains, X, Y, Domain_label, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    
    criterion = LabelSmooth(num_class=args.num_class).to(args.device)
    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        # 2790 for SEED
        # 620 for SEED-IV
        input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9
        hidden_size = 320


        # Initialize the model
        netA = MLPBase(input_size=input_size, hidden_size = hidden_size).to(args.device)
        # netA.apply(init_weights)
        netB = feat_bottleneck(hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim).to(args.device)
        # netB.apply(init_weights)
        netC = feat_classifier(bottleneck_dim=args.bottleneck_dim, class_num=args.num_class).to(args.device)#分类器
        # netC.apply(init_weights)
        netD = feat_classifier(bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2).to(args.device)#领域判别器，目标是获取一个性能强大的领域分类器
        # netD.apply(init_weights)
    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)


    param_group = []
    param_group_A = []
    param_group_B = []
    param_group_C = []
    param_group_D = []
    learning_rate = args.lr_a
    for k, v in netA.named_parameters():
        param_group += [{'params': v,  "lr_mult": 1, 'decay_mult': 2}]
        param_group_A += [{'params': v,  "lr_mult": 1, 'decay_mult': 2}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, "lr_mult": 1, 'decay_mult': 2}]
        param_group_B += [{'params': v,  "lr_mult": 1, 'decay_mult': 2}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, "lr_mult": 1, 'decay_mult': 2}]
        param_group_C += [{'params': v,  "lr_mult": 1, 'decay_mult': 2}]
    for k, v in netD.named_parameters():
        param_group += [{'params': v, "lr_mult": 1, 'decay_mult': 2}]
        param_group_D += [{'params': v,  "lr_mult": 1, 'decay_mult': 2}]

    optimizer_A = torch.optim.SGD(param_group_A, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
    optimizer_B = torch.optim.SGD(param_group_B, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
    optimizer_C = torch.optim.SGD(param_group_C, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)
    optimizer_D = torch.optim.SGD(param_group_D, lr=args.lr_a, momentum=0.9, weight_decay=0.0005)

 
    log_total_loss = []
    final_acc = 0
    mixstyle = None
    if getattr(args, "use_feature_mixstyle", False) and FeatureMixStyle is not None:
        mixstyle = FeatureMixStyle(p=getattr(args, "mixstyle_p", 0.5), alpha=getattr(args, "mixstyle_alpha", 0.1)).to(args.device)
        print(f"[SSAS] FeatureMixStyle enabled in DGMA stage: p={mixstyle.p}, alpha={mixstyle.alpha}")
    rspm = None
    rspm_stats = None
    if getattr(args, "use_rspm", False) and ReliabilitySourcePrototypeMemory is not None:
        rspm = ReliabilitySourcePrototypeMemory(
            num_domains=num_domains - 1,
            num_classes=args.num_class,
            feature_dim=args.bottleneck_dim,
            temperature=getattr(args, "rspm_temperature", 0.2),
            momentum=getattr(args, "rspm_momentum", 0.9),
            target_conf_threshold=getattr(args, "rspm_target_conf_threshold", 0.7),
            reliability_tau=getattr(args, "rspm_reliability_tau", 1.0),
            target_weight=getattr(args, "rspm_target_weight", 0.3),
        ).to(args.device)
        print(
            "[SSAS] RSPM enabled: "
            f"weight={getattr(args, 'rspm_weight', 0.05)}, "
            f"temperature={rspm.temperature}, momentum={rspm.momentum}, "
            f"target_conf_threshold={rspm.target_conf_threshold}, target_weight={rspm.target_weight}"
        )
    ema_modules = None
    ema_consistency_loss = None
    ema_mask = None
    if getattr(args, "use_ema_teacher", False):
        ema_modules = {
            "netA": copy.deepcopy(netA).to(args.device).eval(),
            "netB": copy.deepcopy(netB).to(args.device).eval(),
            "netC": copy.deepcopy(netC).to(args.device).eval(),
        }
        for module in ema_modules.values():
            for param in module.parameters():
                param.requires_grad_(False)
        print(
            "[SSAS] EMA teacher enabled: "
            f"decay={getattr(args, 'ema_decay', 0.99)}, "
            f"weight={getattr(args, 'ema_consistency_weight', 0.05)}, "
            f"conf_threshold={getattr(args, 'ema_conf_threshold', 0.6)}"
        )
    for i in range(args.max_iter2):

        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()
            Dy_src = list()
            # new_shape = (args.batch_size, 62, 9 * 5)
            index = 0
            #列表存储每个源域的批次数据=====================================================这里改特征：切空间特征========================================================
            for domain_idx in range(num_domains - 1):

                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                domain_labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                x_src.append(tmp_x)
                y_src.append(tmp_y)
                Dy_src.append(domain_labels)
                index += 1
            # get the target batch 把验证集拿出来 ,在这里调用augment函数来使得src和trg变成切空间特征    注意，下一步是运行到sugment处，然后查看且空间的维度，来改变神经网络的降维数
            #主要是消极和中性的类别无法区分，那么就需要采用一些特殊的手段：比如在第一阶段的训练中将happy分类拿走，包括目标域中被分为happy的数据，然后对其他数据再进行重新训练测试
            #问题在于，如何确定所有类别中：哪两种类别易于混淆呢？怎么获取最差的两类？
            #w问题在于：如何在预训练后将混淆的两类区分开 第一种做法是：CSP来扩大两者的差距？==》可用cspNet
            x_trg = data['Tx'].float().cuda()
            # x_trgg = x_trg.view(*new_shape)
            # x_trg = augment(x_trgg).cuda()
            # Enable model to train
            netA.train(True)
            netB.train(True)
            netC.train(True)
            netD.train(True)
            netF.train(True)


            optimizer_A = lr_schedule.inv_lr_scheduler(optimizer_A, i, lr=args.lr_a)
            optimizer_B = lr_schedule.inv_lr_scheduler(optimizer_B, i, lr=args.lr_a)
            optimizer_C = lr_schedule.inv_lr_scheduler(optimizer_C, i, lr=args.lr_a)
            optimizer_D = lr_schedule.inv_lr_scheduler(optimizer_D, i, lr=args.lr_a)            

            features_target = netB(netA(x_trg))
            outputs_target = netC(features_target)
            if ema_modules is not None:
                with torch.no_grad():
                    ema_features_target = ema_modules["netB"](ema_modules["netA"](x_trg))
                    ema_outputs_target = ema_modules["netC"](ema_features_target)
                    ema_probs = torch.nn.functional.softmax(ema_outputs_target, dim=1)
                    ema_conf, _ = ema_probs.max(dim=1)
                    ema_mask = ema_conf >= getattr(args, "ema_conf_threshold", 0.6)
                if ema_mask.any():
                    ema_consistency_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(outputs_target[ema_mask], dim=1),
                        ema_probs[ema_mask],
                        reduction="batchmean",
                    )
                else:
                    ema_consistency_loss = outputs_target.new_tensor(0.0)
            else:
                ema_consistency_loss = None
                ema_mask = None
            # 目标域预测标签
            pseu_labels_target = torch.argmax(outputs_target, dim=1)


            pred_src_domain_D = []
            pred_src_domain_F = []
            pred_src_class = []
            pred_src = []
            rspm_source_features = []
            rspm_source_labels = []
            coral_loss = 0
            mmd_b_loss = 0 
            mmd_t_loss = 0
            loss_lmmd_2 = 0
            for domain_idx in range(num_domains  - 1):
    
                features_source = netB(netA(x_src[domain_idx]))
                if mixstyle is not None:
                    features_source = mixstyle(features_source)
                features_s_Adver = Adver_network.ReverseLayerF.apply(features_source, args.gamma)#用这个替代features_source经过了反转层
                outputs_source_domain_D = netD(features_s_Adver)
                outputs_source_domain_F = netF(features_source)
                output_source_class = netC(features_source)
                pred_src_domain_D.append(outputs_source_domain_D)
                pred_src_domain_F.append(outputs_source_domain_F)
                pred_src_class.append(output_source_class)
                rspm_source_features.append(features_source)
                rspm_source_labels.append(y_src[domain_idx])
                # coral_loss = utils.CORAL_loss(features_source, features_target)
                mmd_b_loss += utils.marginal(features_source,features_target)
                mmd_t_loss += utils.conditional(
                    features_source,
                    features_target,
                    y_src[domain_idx].reshape((args.batch_size, 1)),
                    torch.nn.functional.softmax(outputs_target,dim = 1),
                    2.0,
                    5,
                    None)
            # 将每个源域的标签拼接起来
            pred_source_domain_D = torch.cat(pred_src_domain_D, dim=0)
            pred_source_domain_F = torch.cat(pred_src_domain_F, dim=0)
            pred_source_class = torch.cat(pred_src_class, dim=0)
            labels_source = torch.cat(y_src, dim=0)
            Domain_labels_source = torch.cat(Dy_src, dim=0)
            # 交叉熵损失
            # classifier_loss = nn.CrossEntropyLoss()(pred_source, labels_source)
            classifier_loss = criterion(pred_source_class, labels_source.flatten())
            Adver_domain_labels_loss = criterion(pred_source_domain_D, Domain_labels_source.flatten())
            same_domain_loss = discrepancy(pred_source_domain_D,pred_source_domain_F)

            
            #[MMD loss]===================================================这里改损失：MMD损失，Wasserstein损失，对抗损失等===========================================================================
            
            MMD_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss
            # MMD_loss = loss_lmmd_2
            total_loss = classifier_loss + Adver_domain_labels_loss + MMD_loss + same_domain_loss #一个交叉熵加上CMD、SM的领域自适应损失，再加上一个目标域的损失
            if rspm is not None:
                rspm_loss, rspm_stats = rspm(
                    source_features=rspm_source_features,
                    source_labels=rspm_source_labels,
                    target_features=features_target,
                    target_logits=outputs_target,
                )
                total_loss = total_loss + getattr(args, "rspm_weight", 0.05) * rspm_loss
            if ema_consistency_loss is not None:
                total_loss = total_loss + getattr(args, "ema_consistency_weight", 0.05) * ema_consistency_loss
            if getattr(args, "use_target_entropy", False):
                target_entropy_loss = utils.Entropylogits(outputs_target)
                total_loss = total_loss + getattr(args, "target_entropy_weight", 0.01) * target_entropy_loss
            else:
                target_entropy_loss = None

            # 重置梯度
            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            optimizer_C.zero_grad()
            optimizer_D.zero_grad()

            # Compute gradients
            # [normal]
            total_loss.backward()

            # [Update weights]
            # classifier
            # optimizer.step()
            optimizer_A.step()
            optimizer_B.step()
            optimizer_C.step()
            optimizer_D.step()
            if ema_modules is not None:
                _update_ema_module(ema_modules["netA"], netA, getattr(args, "ema_decay", 0.99))
                _update_ema_module(ema_modules["netB"], netB, getattr(args, "ema_decay", 0.99))
                _update_ema_module(ema_modules["netC"], netC, getattr(args, "ema_decay", 0.99))
        # 模型转变test
        netA.train(False)
        netB.train(False)
        netC.train(False)
        netD.train(False)
        final_f1 = 0
        final_auc = 0
        final_mat =[]
        # 计算准确率及其他指标
        acc, best_f1, best_auc, best_mat, features, labels = test_muda(dataset_test, netA,netB,netC,args)
        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, acc, best_f1, best_auc)
        if rspm_stats is not None:
            log_str += (
                " \t rspm_loss: {:.4f} \t rspm_src: {} \t rspm_tgt: {} "
                "\t rel_mean: {:.4f} \t center_norm: {:.4f}"
            ).format(
                rspm_stats["rspm_loss"],
                rspm_stats["rspm_valid_source_samples"],
                rspm_stats["rspm_valid_target_samples"],
                rspm_stats["rspm_reliability_mean"],
                rspm_stats["rspm_center_norm"],
            )
        if ema_consistency_loss is not None:
            log_str += " \t ema_loss: {:.4f} \t ema_valid: {}".format(
                float(ema_consistency_loss.detach().item()),
                int(ema_mask.sum().item()) if ema_mask is not None else 0,
            )
        if target_entropy_loss is not None:
            log_str += " \t target_entropy: {:.4f}".format(float(target_entropy_loss.detach().item()))
        if final_acc < acc:
            final_acc = acc
            final_f1 = best_f1
            final_auc = best_auc
            final_mat = best_mat
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, Y, final_acc, final_f1, final_auc, final_mat,  log_total_loss, acc


@torch.no_grad()
def _update_ema_module(teacher, student, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)
