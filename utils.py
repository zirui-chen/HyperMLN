# -- coding: utf-8 --

def mln_tuple_prob(tuple_, tuple2p):
    if tuple(tuple_) in tuple2p:
        return tuple2p[tuple(tuple_)]
    return 0.5

def evaluate(mln_pred_file, khge_pred_file_raw, khge_pred_file_fil, output_file_raw, output_file_fil, weight):
    hit1_raw = 0
    hit1_fil = 0
    hit3_raw = 0
    hit3_fil = 0
    hit10_raw = 0
    hit10_fil = 0
    mr_raw = 0
    mr_fil = 0
    mrr_raw = 0
    mrr_fil = 0
    cn = 0
    tuple2p = dict()
    with open(mln_pred_file, 'r') as f:
        for line in f:
            l = line.strip().split("\t")
            tuple_ = tuple(l[:-1])
            p = float(l[-1])
            tuple2p[tuple_] = p

        f.close()

    with open(khge_pred_file_raw, "r") as f1, open(khge_pred_file_fil, "r") as f2:
        while True:
            truth_raw = f1.readline()
            preds_raw = f1.readline()
            truth_fil = f2.readline()
            preds_fil = f2.readline()

            if (not truth_raw) or (not preds_raw) or (not truth_fil) or (not preds_fil):
                break

            truth_raw = truth_raw.strip().split()
            preds_raw = preds_raw.strip().split()
            truth_fil = truth_fil.strip().split()
            preds_fil = preds_fil.strip().split()

            tuple_raw = truth_raw[:-2]
            mode_raw = truth_raw[-2]
            original_ranking_raw = int(truth_raw[-1])

            tuple_fil = truth_fil[:-2]
            mode_fil = truth_fil[-2]
            original_ranking_fil = int(truth_fil[-1])

            if mode_raw == "pos1":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e1 = preds_raw[k][0]
                    new_tuple_raw = [tuple_raw[0]] + [e1] + tuple_raw[2:]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e1 = preds_raw[k][0]
                    if e1 == tuple_raw[1]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e1 = preds_fil[k][0]
                    new_tuple_fil = [tuple_fil[0]] + [e1] + tuple_fil[2:]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e1 = preds_fil[k][0]
                    if e1 == tuple_fil[1]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            elif mode_raw == "pos2":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e2 = preds_raw[k][0]
                    new_tuple_raw = tuple_raw[:2] + [e2] + tuple_raw[3:]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e2 = preds_raw[k][0]
                    if e2 == tuple_raw[2]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e2 = preds_fil[k][0]
                    new_tuple_fil = tuple_fil[:2] + [e2] + tuple_fil[3:]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e2 = preds_fil[k][0]
                    if e2 == tuple_fil[2]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            elif mode_raw == "pos3":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e3 = preds_raw[k][0]
                    new_tuple_raw = tuple_raw[:3] + [e3] + tuple_raw[4:]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e3 = preds_raw[k][0]
                    if e3 == tuple_raw[3]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e3 = preds_fil[k][0]
                    new_tuple_fil = tuple_fil[:3] + [e3] + tuple_fil[4:]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e3 = preds_fil[k][0]
                    if e3 == tuple_fil[3]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            elif mode_raw == "pos4":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e4 = preds_raw[k][0]
                    new_tuple_raw = tuple_raw[:4] + [e4] + tuple_raw[5:]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e4 = preds_raw[k][0]
                    if e4 == tuple_raw[4]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e4 = preds_fil[k][0]
                    new_tuple_fil = tuple_fil[:4] + [e4] + tuple_fil[5:]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e4 = preds_fil[k][0]
                    if e4 == tuple_fil[4]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            elif mode_raw == "pos5":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e5 = preds_raw[k][0]
                    new_tuple_raw = tuple_raw[:5] + [e5] + tuple_raw[6:]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e5 = preds_raw[k][0]
                    if e5 == tuple_raw[5]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e5 = preds_fil[k][0]
                    new_tuple_fil = tuple_fil[:5] + [e5] + tuple_fil[6:]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e5 = preds_fil[k][0]
                    if e5 == tuple_fil[5]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            elif mode_raw == "pos6":
                preds_raw = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_raw]
                for k in range(len(preds_raw)):
                    e6 = preds_raw[k][0]
                    new_tuple_raw = tuple_raw[:6] + [e6]
                    preds_raw[k][1] += mln_tuple_prob(new_tuple_raw, tuple2p) * weight

                preds_raw = sorted(preds_raw, key=lambda x: x[1], reverse=True)
                ranking_raw = -1
                for k in range(len(preds_raw)):
                    e6 = preds_raw[k][0]
                    if e6 == tuple_raw[6]:
                        ranking_raw = k + 1
                        break
                if ranking_raw == -1:
                    ranking_raw = original_ranking_raw

                preds_fil = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds_fil]
                for k in range(len(preds_fil)):
                    e6 = preds_fil[k][0]
                    new_tuple_fil = tuple_fil[:6] + [e6]
                    preds_fil[k][1] += mln_tuple_prob(new_tuple_fil, tuple2p) * weight

                preds_fil = sorted(preds_fil, key=lambda x: x[1], reverse=True)
                ranking_fil = -1
                for k in range(len(preds_fil)):
                    e6 = preds_fil[k][0]
                    if e6 == tuple_fil[6]:
                        ranking_fil = k + 1
                        break
                if ranking_fil == -1:
                    ranking_fil = original_ranking_fil

            if ranking_raw <= 1:
                hit1_raw += 1
            if ranking_raw <=3:
                hit3_raw += 1
            if ranking_raw <= 10:
                hit10_raw += 1

            mr_raw += ranking_raw
            mrr_raw += 1.0 / ranking_raw

            if ranking_fil <= 1:
                hit1_fil += 1
            if ranking_fil <=3:
                hit3_fil += 1
            if ranking_fil <= 10:
                hit10_fil += 1

            mr_fil += ranking_fil
            mrr_fil += 1.0 / ranking_fil

            cn += 1

    mr_raw /= cn
    mrr_raw /= cn
    hit1_raw /= cn
    hit3_raw /= cn
    hit10_raw /= cn

    print('MR_RAW: ', mr_raw)
    print('MRR_RAW: ', mrr_raw)
    print('Hit@1_RAW: ', hit1_raw)
    print('Hit@3_RAW: ', hit3_raw)
    print('Hit@10_RAW: ', hit10_raw)

    mr_fil /= cn
    mrr_fil /= cn
    hit1_fil /= cn
    hit3_fil /= cn
    hit10_fil /= cn

    print('MR_FIL: ', mr_fil)
    print('MRR_FIL: ', mrr_fil)
    print('Hit@1_FIL: ', hit1_fil)
    print('Hit@3_FIL: ', hit3_fil)
    print('Hit@10_FIL: ', hit10_fil)

    with open(output_file_raw, "w") as f1, open(output_file_fil, "w") as f2:
        f1.write("MR: {}\n".format(mr_raw))
        f1.write("MRR: {}\n".format(mrr_raw))
        f1.write("Hit@1: {}\n".format(hit1_raw))
        f1.write("Hit@3: {}\n".format(hit3_raw))
        f1.write("Hit@10:{}\n".format(hit10_raw))
        f2.write("MR: {}\n".format(mr_fil))
        f2.write("MRR: {}\n".format(mrr_fil))
        f2.write("Hit@1: {}\n".format(hit1_fil))
        f2.write("Hit@3: {}\n".format(hit3_fil))
        f2.write("Hit@10:{}\n".format(hit10_fil))




def augment_tuple(pred_file, tuple_file, out_file, threshold):
    with open(pred_file, 'r') as f:
        data = []
        for line in f:
            l = line.strip().split("\t")
            data += [tuple(l)]
        f.close()

    with open(tuple_file, 'r') as f:
        tuples = set()
        for line in f:
            l = line.strip().split("\t")
            tuples.add(tuple(l))

        for tp in data:
            if float(tp[-1]) < threshold:
                continue
            tuples.add(tuple(tp[:-1]))
        f.close()

    with open(out_file, 'w') as f:
        for tuple_ in tuples:
            for ele in tuple_[:-1]:
                f.write('{}\t'.format(ele))
            f.write('{}\n'.format(tuple_[-1]))
        f.close()