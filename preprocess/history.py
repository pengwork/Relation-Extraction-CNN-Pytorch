# bert_params_list = list(map(id, model.word_embedding.parameters()))
# bert_parms = filter(lambda p: id(p) in bert_params_list and p.requires_grad,
#                     model.parameters())
#
# pos_params_list = list(map(id, model.pos1_embedding.parameters())) + list(
#     map(id, model.pos2_embedding.parameters()))
# pos_params = filter(lambda p: id(p) in pos_params_list and p.requires_grad,
#                     model.parameters())
#
# base_params = filter(lambda p: id(p) not in bert_params_list + pos_params_list and p.requires_grad,
#                      model.parameters())
# optimizer = optim.Adam([{'params': bert_parms, 'lr': config.lr_bert},
#                         {'params': pos_params, 'lr': config.lr},
#                         {'params': base_params, 'lr': config.lr}], weight_decay=config.weight_decay)