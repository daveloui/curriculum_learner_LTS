import sys
import os
import os.path as path
from os import listdir
from os.path import isfile, join

import argparse
import pickle
import copy
import random
import math
from typing import Dict, Any
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Witness_Puzzle_Image import WitnessPuzzle


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    return objects


flatten_list = lambda l: [item for sublist in l for item in sublist]


def plot_data(list_values, idxs_new_puzzles, title_name, filename_to_save_fig, special_x, special_y,
              x_label="Number of Puzzles Solved", y_lim_upper=None, y_lim_lower=None, x_max=2375, x_min=2244):
    plt.figure ()
    for xc in idxs_new_puzzles:
        plt.axvline (x=xc, c='orange', linestyle='--', alpha=0.6)
    x = np.arange(0, 2369)
    list_values = np.asarray(list_values)
    plt.scatter (x, list_values, s=4, alpha=1.0, color='b')

    plt.grid (True)
    plt.title(title_name)
    plt.xlabel(x_label)
    if y_lim_upper is not None and y_lim_lower is not None:
        plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    if x_max is not None and x_min is not None:
        plt.xlim(xmin=x_min, xmax=x_max)

    # plt.show()
    plt.savefig (filename_to_save_fig)
    # plt.close ()

    print(title_name)
    if "Levin Cost" in title_name:
        zoomed_title = "Zoomed Levin Cost"
    elif "Cosines" in title_name:
        zoomed_title = "Zoomed " + title_name
    else:
        zoomed_title = "Zoomed " + title_name

    string_list = filename_to_save_fig.split ("plots/")
    zoomed_filename_to_save_fig = string_list[0] + "plots/" + "Zoom_" + string_list[1]
    print("zoomed_filename_to_save_fig", zoomed_filename_to_save_fig)
    plt.xlim(xmin=2200, xmax=2375)
    plt.title (zoomed_title)
    # plt.show()
    # plt.savefig (zoomed_filename_to_save_fig)
    plt.close()

    return


def get_witness_ordering(flat_list, d):
    witness_ord = []
    for tup in flat_list:
        if "wit" in tup[0]:
            witness_ord.append (tup)
    d[file] = witness_ord
    return d


def separate_names_and_vals(flat_list):
    list_names = []
    list_vals = []
    for tup in flat_list:
        name = tup[0]
        val = tup[1]
        list_names += [name]
        list_vals += [val]
    return list_names, list_vals


def find_special_vals(loaded_object):
    # for sublist in loaded_object[0]:
    #     pass
    return None, None


def map_witness_puzzles_to_dims(name):
    if (name == "witness_1") or (name == "witness_2"):
        return "1x2"
    elif name == "witness_3":
        return "1x3"
    elif name == "witness_4":
        return "2x2"
    elif (name == "witness_5") or (name == "witness_6"):
        return "3x3"
    elif (name == "witness_7") or (name == "witness_8") or (name == "witness_9"):
        return "4x4"


print("Idxs_rank_data_BFS_4x4.pkl")
idx_object = open_pickle_file("puzzles_4x4_theta_n-theta_i/Idxs_rank_data_BFS_4x4.pkl")
print("len(idx_object[0])", len(idx_object[0]))
print("")

puzzles_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4/puzzle_imgs")
print ("puzzle images path =", puzzles_path)
if not os.path.exists (puzzles_path):
    os.makedirs (puzzles_path, exist_ok=True)

plots_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4/plots")
print ("plots_path =", plots_path)
if not os.path.exists (plots_path):
    os.makedirs (plots_path, exist_ok=True)

Rank_DotProds = [('4x4_283', 1.3617835064248265), ('4x4_75', 1.3127930131774395), ('4x4_229', 1.2937882431028216), ('4x4_816', 1.189628144607668), ('4x4_593', 1.1837613982239519), ('4x4_139', 1.1248946030594076), ('4x4_567', 1.0659011853054665), ('4x4_591', 1.0529664000335657), ('4x4_680', 1.0483880960060863), ('4x4_594', 1.0436179252567508), ('4x4_509', 1.0362324399177845), ('4x4_225', 1.0335518400880774), ('4x4_383', 1.0295327250298136), ('4x4_695', 1.011494864938927), ('4x4_156', 1.0027173806218952), ('4x4_83', 1.0018666867870014), ('4x4_180', 0.9726639929251818), ('4x4_412', 0.9354994713165579), ('4x4_329', 0.8988621285535318), ('4x4_238', 0.8939739275881762), ('4x4_240', 0.8580054524755545), ('4x4_635', 0.8267372285430686), ('4x4_486', 0.8235548700937048), ('4x4_698', 0.8077417263599869), ('4x4_385', 0.7716546991328429), ('4x4_273', 0.7704125314306667), ('4x4_981', 0.7550184634245454), ('4x4_565', 0.752173182142222), ('4x4_115', 0.7231386381345943), ('4x4_759', 0.714349408972465), ('4x4_15', 0.7121191620006664), ('4x4_406', 0.7055801456312587), ('4x4_473', 0.7035412286190468), ('4x4_643', 0.6914463635986122), ('4x4_799', 0.6859691772425797), ('4x4_553', 0.6859347549022418), ('4x4_308', 0.6844633108852243), ('4x4_247', 0.6771207882147781), ('4x4_278', 0.6730085108834729), ('4x4_921', 0.6716294585645792), ('4x4_752', 0.6687794949924193), ('4x4_839', 0.663609945295783), ('4x4_779', 0.6569610667232826), ('4x4_80', 0.6524358387182175), ('4x4_439', 0.6429415860212542), ('4x4_610', 0.6381960459266266), ('4x4_908', 0.6232030145663776), ('4x4_282', 0.611787919213083), ('4x4_794', 0.5945369308243255), ('4x4_413', 0.5707620486448512), ('4x4_245', 0.5678953347407084), ('4x4_427', 0.5657492824458775), ('4x4_829', 0.565100048330344), ('4x4_564', 0.5412133848609104), ('4x4_186', 0.5387145438166814), ('4x4_927', 0.5343591487723399), ('4x4_757', 0.5179127294775775), ('4x4_12', 0.5123104627601596), ('4x4_430', 0.506230695457344), ('4x4_360', 0.5019959881085142), ('4x4_551', 0.49771547328037674), ('4x4_701', 0.4869402967531473), ('4x4_667', 0.4769180635298809), ('4x4_38', 0.47605924597486027), ('4x4_71', 0.4741848801970634), ('3x3_662', 0.46954642369877375), ('4x4_64', 0.46898348581238664), ('4x4_905', 0.4682285048899138), ('4x4_988', 0.46767166261939913), ('4x4_453', 0.46145738846580425), ('4x4_481', 0.44480948823777605), ('4x4_338', 0.43734080371395934), ('4x4_162', 0.43552163828397267), ('4x4_845', 0.4347565755663983), ('4x4_975', 0.42567741063226006), ('4x4_206', 0.41948352871878614), ('4x4_529', 0.4180527255387139), ('4x4_494', 0.41465219036932), ('4x4_709', 0.40955707044126444), ('4x4_630', 0.4087178204320785), ('4x4_725', 0.406196977030133), ('4x4_778', 0.40295753307742205), ('4x4_999', 0.400017160071856), ('4x4_877', 0.39766316123258677), ('4x4_844', 0.3960622946197392), ('4x4_423', 0.386357877989276), ('4x4_569', 0.3790700177183871), ('4x4_56', 0.3775415778627428), ('4x4_179', 0.37000671920112627), ('witness_8', 0.36818614658548754), ('4x4_284', 0.3600177223920978), ('4x4_479', 0.359816946018741), ('4x4_488', 0.35924851975807265), ('4x4_344', 0.35921022876057895), ('4x4_210', 0.3559516317438167), ('4x4_155', 0.35534708457049163), ('4x4_187', 0.34988975444188364), ('4x4_464', 0.3497591001460466), ('4x4_8', 0.3486552141247826), ('4x4_628', 0.34752354682551856), ('4x4_964', 0.3474094174249197), ('4x4_130', 0.34715205082799067), ('4x4_847', 0.34624484473599776), ('4x4_324', 0.34343542923836545), ('4x4_281', 0.3423013484245038), ('4x4_175', 0.3396740140022962), ('4x4_714', 0.3385656115278247), ('4x4_557', 0.337841076484757), ('4x4_45', 0.3305511652737052), ('4x4_764', 0.3293742382459751), ('4x4_694', 0.328721214138471), ('4x4_853', 0.322787135451913), ('4x4_116', 0.30884987495143684), ('4x4_81', 0.3068914810487279), ('4x4_402', 0.2978007744737833), ('4x4_326', 0.2942958512497177), ('4x4_724', 0.29427598735919247), ('4x4_394', 0.29308184791170255), ('4x4_631', 0.2901835754441782), ('4x4_935', 0.2875633145954989), ('4x4_190', 0.287501323300432), ('4x4_193', 0.28465375876954535), ('4x4_96', 0.2845828475922711), ('4x4_297', 0.28299566654286656), ('4x4_536', 0.27693809433175604), ('4x4_140', 0.276683033669865), ('4x4_876', 0.27514370281201883), ('4x4_974', 0.2743449476821333), ('4x4_769', 0.27267463712980206), ('4x4_93', 0.2724771967477377), ('4x4_220', 0.26521065097211777), ('4x4_889', 0.2627426314634796), ('4x4_456', 0.25809686154111633), ('4x4_502', 0.25797267012351743), ('4x4_671', 0.2509660559015935), ('4x4_356', 0.2497515579105603), ('4x4_135', 0.248218179682409), ('4x4_747', 0.2464637434678041), ('4x4_934', 0.24365807757613855), ('4x4_503', 0.23845135316710908), ('4x4_568', 0.23517266574378692), ('4x4_666', 0.23340960157242038), ('4x4_672', 0.23224679009488297), ('4x4_984', 0.23187216820143167), ('4x4_22', 0.23039304147541959), ('4x4_444', 0.22803053722467626), ('4x4_51', 0.22625034246381379), ('4x4_733', 0.2255058759098842), ('4x4_105', 0.22459701971087953), ('4x4_880', 0.22359178319114145), ('4x4_979', 0.22108560395633906), ('4x4_471', 0.21994278308374365), ('4x4_357', 0.2188140334451385), ('4x4_28', 0.21840854939933335), ('4x4_741', 0.2140342212260384), ('4x4_334', 0.2087503412562174), ('4x4_294', 0.20766421924959128), ('4x4_797', 0.20649917949512925), ('4x4_107', 0.2061285229321076), ('4x4_796', 0.20455791963658385), ('4x4_32', 0.20453785877082833), ('4x4_575', 0.20394232799802042), ('4x4_803', 0.20167068939941765), ('4x4_95', 0.20133868414198708), ('4x4_590', 0.20111940744538456), ('4x4_862', 0.1972035183552872), ('4x4_251', 0.1969830433358287), ('4x4_143', 0.19491797176228293), ('4x4_449', 0.19088772150067806), ('4x4_931', 0.19029020363181814), ('4x4_677', 0.18960238972666615), ('4x4_380', 0.1882241660092112), ('4x4_860', 0.1880822772532922), ('4x4_131', 0.18402947970827754), ('4x4_750', 0.18235497749914475), ('4x4_907', 0.1819545527684324), ('4x4_350', 0.18034685796189912), ('4x4_801', 0.17813362422507606), ('4x4_617', 0.1770109225016464), ('4x4_602', 0.17689925448022548), ('4x4_654', 0.17483141856025947), ('4x4_498', 0.17421582271482705), ('4x4_955', 0.17309851572705193), ('4x4_242', 0.17297370023571748), ('4x4_638', 0.1689870122825001), ('4x4_598', 0.16707498839668988), ('4x4_522', 0.1665086496017942), ('4x4_391', 0.16415108813145526), ('4x4_77', 0.16173646963351884), ('4x4_425', 0.15883057572954123), ('4x4_648', 0.15259756604522576), ('4x4_904', 0.151333398472316), ('4x4_763', 0.14626672557795678), ('4x4_287', 0.14578706949576925), ('4x4_381', 0.14325016011400873), ('4x4_44', 0.14309896940896968), ('4x4_746', 0.14246380076867776), ('4x4_644', 0.14234789693318411), ('4x4_511', 0.1400171509780458), ('4x4_232', 0.13973418818055872), ('4x4_111', 0.1390985295513038), ('4x4_589', 0.13837721570831818), ('4x4_386', 0.13701512547542877), ('4x4_879', 0.13633709852536674), ('4x4_851', 0.1348826725735745), ('4x4_811', 0.13327789547216118), ('4x4_367', 0.1316675068809619), ('4x4_945', 0.1315189269465832), ('4x4_400', 0.128681002568478), ('4x4_852', 0.12845888394755672), ('4x4_820', 0.12771115813185263), ('4x4_253', 0.12625493728095438), ('4x4_874', 0.12609462406691704), ('4x4_504', 0.12544067299677034), ('4x4_983', 0.12524456184743676), ('4x4_300', 0.12476159493307265), ('4x4_525', 0.12285189723796583), ('4x4_562', 0.12236661842437427), ('4x4_85', 0.12084245747585912), ('4x4_148', 0.11988189500107546), ('4x4_962', 0.11549508524478604), ('4x4_857', 0.11531956902543239), ('4x4_993', 0.11399244087250915), ('4x4_112', 0.11366002602099497), ('4x4_641', 0.11313457877018579), ('4x4_521', 0.11109754392178584), ('4x4_306', 0.10669473043422235), ('4x4_114', 0.10547496461370662), ('4x4_613', 0.10517166858663654), ('4x4_687', 0.10453815269447407), ('4x4_838', 0.10231186237114262), ('4x4_34', 0.10223007016136725), ('4x4_911', 0.10099006659529185), ('4x4_409', 0.10002384127818607), ('4x4_550', 0.09777354596187689), ('4x4_207', 0.09743141401403904), ('4x4_261', 0.09378629187365342), ('4x4_24', 0.09266878038098716), ('4x4_198', 0.09238318990301495), ('4x4_910', 0.08847467232316147), ('4x4_237', 0.08777048892878461), ('4x4_332', 0.0877421800264416), ('4x4_354', 0.08718050130863587), ('4x4_579', 0.08521912652441117), ('4x4_268', 0.0830599632764605), ('4x4_863', 0.08233946097716498), ('4x4_4', 0.08093763963886845), ('4x4_915', 0.07832562741706513), ('4x4_916', 0.07674369093346037), ('4x4_345', 0.0763336098906793), ('4x4_740', 0.0755142727879917), ('4x4_669', 0.07379629703053071), ('4x4_760', 0.0731109975886), ('4x4_270', 0.07208897399967858), ('4x4_25', 0.07076682301011772), ('4x4_248', 0.0699693054796103), ('4x4_374', 0.06877489514269113), ('4x4_395', 0.0686100036883503), ('4x4_465', 0.06823683483254607), ('4x4_161', 0.06665598154783708), ('4x4_603', 0.06632708441129825), ('4x4_403', 0.06548309256362933), ('4x4_817', 0.06537953212322806), ('4x4_580', 0.06393481675216173), ('4x4_327', 0.06380500344450944), ('4x4_482', 0.06315339045728599), ('4x4_966', 0.062476202116507416), ('4x4_168', 0.06086118529290556), ('4x4_956', 0.05917059669463335), ('4x4_780', 0.05791469848166005), ('4x4_989', 0.05743679197307026), ('4x4_359', 0.05736791879720978), ('4x4_922', 0.057293588478907455), ('4x4_98', 0.05703791116040209), ('4x4_751', 0.05599087711153138), ('4x4_176', 0.0559009468854713), ('4x4_828', 0.05091871576868598), ('4x4_960', 0.05016686711694584), ('4x4_317', 0.049931834647569735), ('4x4_773', 0.04986906689814994), ('4x4_2', 0.048788743604198895), ('4x4_231', 0.047715834179784054), ('4x4_682', 0.04737660538512256), ('4x4_215', 0.04703743516358306), ('4x4_200', 0.046404900330596334), ('4x4_636', 0.04629492776840492), ('4x4_642', 0.046054356704098715), ('4x4_432', 0.04592322942665462), ('4x4_262', 0.04583151788913877), ('4x4_337', 0.041678969713235654), ('4x4_992', 0.041598270557802075), ('4x4_209', 0.04136090547687582), ('4x4_578', 0.041085934625181786), ('4x4_761', 0.03982919098775261), ('4x4_194', 0.038766099474233075), ('4x4_821', 0.03847380091290547), ('4x4_492', 0.037942582688025965), ('4x4_826', 0.03712428072773814), ('4x4_537', 0.03701279442206911), ('4x4_539', 0.03565673661560183), ('4x4_620', 0.035343407961487334), ('4x4_675', 0.03452738881739127), ('4x4_929', 0.03445491110831097), ('4x4_275', 0.033743951825042764), ('4x4_446', 0.03281613455518253), ('4x4_637', 0.03272371542045932), ('4x4_861', 0.03127190519187959), ('4x4_387', 0.030274572250155632), ('4x4_419', 0.030023519413977698), ('4x4_352', 0.029119650995444977), ('4x4_549', 0.02904811646666252), ('4x4_450', 0.028800151989696876), ('4x4_791', 0.028391301275538957), ('4x4_692', 0.027712090332386745), ('4x4_373', 0.026953304121049633), ('4x4_35', 0.025456406489384815), ('4x4_417', 0.02540934787889614), ('4x4_246', 0.02536736355037189), ('4x4_976', 0.02456650657351682), ('4x4_320', 0.023878989161116713), ('4x4_835', 0.023799214572755294), ('4x4_681', 0.023717292364810395), ('4x4_753', 0.02335181275255617), ('4x4_84', 0.022599229085256446), ('4x4_363', 0.02187634962255449), ('4x4_720', 0.020758447746392582), ('4x4_46', 0.020666739668422203), ('4x4_533', 0.01986649567217937), ('4x4_684', 0.01963368366914734), ('4x4_679', 0.018808170453257438), ('4x4_461', 0.018025429578537647), ('4x4_743', 0.015777147230763434), ('4x4_18', 0.013584719965460568), ('4x4_134', 0.01281598219440754), ('4x4_440', 0.012642315287878535), ('4x4_640', 0.012219665400362869), ('4x4_104', 0.00989721365859906), ('4x4_885', 0.005891134316988553), ('4x4_807', 0.004313672727352417), ('4x4_399', 0.0035810690395818618), ('4x4_43', 0.0035526700656586018), ('4x4_822', 0.003124420483553071), ('4x4_581', 0.0030926209273240326), ('4x4_491', 0.0026958438190884822), ('3x3_290', 0.0008989972658291275), ('4x4_881', 0.00020751131024734568), ('4x4_11', -0.00025458962037191047), ('4x4_526', -0.0003618814593635191), ('4x4_216', -0.0012883313551590022), ('4x4_512', -0.003855962041165288), ('4x4_928', -0.005221604467319527), ('4x4_472', -0.005434218194119543), ('4x4_998', -0.008819205649954701), ('4x4_152', -0.009305581136130617), ('4x4_924', -0.009741222170955991), ('4x4_355', -0.011270157855508046), ('4x4_606', -0.013634297168817235), ('4x4_782', -0.014105409231431193), ('4x4_623', -0.014841247955683853), ('4x4_690', -0.015097461491330748), ('4x4_227', -0.015433331468624994), ('4x4_372', -0.016022321798128535), ('4x4_572', -0.017450848310196092), ('4x4_570', -0.01874496554612324), ('4x4_588', -0.019014835741717348), ('4x4_442', -0.021050938700990604), ('4x4_887', -0.022287904206843987), ('4x4_563', -0.023336425251982366), ('4x4_138', -0.024142172782748623), ('4x4_708', -0.024899259781355064), ('4x4_882', -0.024994771502435145), ('4x4_869', -0.026472906783699346), ('4x4_996', -0.027058064034276207), ('4x4_133', -0.03326838509015516), ('4x4_997', -0.03488193485784555), ('4x4_618', -0.035777973786356165), ('4x4_377', -0.041936547732627116), ('4x4_263', -0.044068664942443256), ('4x4_73', -0.0510194204546825), ('4x4_260', -0.05187437855009611), ('4x4_313', -0.05797293297772792), ('4x4_441', -0.06001243425538365), ('4x4_674', -0.06031937056585725), ('4x4_516', -0.06308497859287941), ('4x4_204', -0.06526554997595407), ('4x4_547', -0.06607182371765202), ('4x4_228', -0.06826741036295517), ('4x4_584', -0.06959972286586984), ('4x4_903', -0.07290387446482588), ('4x4_347', -0.07444754954461152), ('4x4_715', -0.0846659210602736), ('4x4_328', -0.08920327493835477), ('4x4_312', -0.09334516047697367), ('4x4_299', -0.1066357605574231), ('4x4_119', -0.11299742989406816), ('4x4_685', -0.11372391796939113), ('4x4_968', -0.1203289910407907), ('4x4_969', -0.13191085240116993), ('4x4_925', -0.14542664335302963), ('4x4_490', -0.1513499437745567), ('4x4_854', -0.1688916842522195), ('4x4_154', -0.1827282047681088), ('4x4_774', -0.1888128910244139), ('4x4_127', -0.19471075780600208), ('4x4_609', -0.21857728441784002), ('4x4_830', -0.2545605206761791), ('4x4_6', -0.2721340332466806), ('4x4_27', -0.3610219162131204), ('4x4_768', -0.38274387765721474), ('4x4_582', -0.4416110285080264)]
Rank_Cosines = [('4x4_75', 0.053513762030853555), ('4x4_486', 0.044384027879720095), ('4x4_83', 0.04036220893222764), ('4x4_695', 0.03968149026767787), ('4x4_757', 0.039528833680454026), ('4x4_816', 0.03949612767526724), ('4x4_635', 0.0390000796164302), ('4x4_180', 0.038745783788987985), ('4x4_567', 0.03790234824793864), ('4x4_981', 0.0376730609313834), ('4x4_229', 0.03636470065789625), ('4x4_509', 0.0359840013091551), ('4x4_283', 0.03596536134035569), ('4x4_412', 0.03526769242695515), ('4x4_345', 0.034546302210931336), ('4x4_502', 0.03412758008761242), ('4x4_156', 0.03294815991943392), ('4x4_282', 0.032608045377959964), ('4x4_844', 0.03193054695904698), ('4x4_628', 0.030897074063734847), ('4x4_135', 0.030733735131256445), ('4x4_80', 0.03002957019784947), ('4x4_186', 0.03002284979185528), ('4x4_273', 0.029266163885901728), ('4x4_680', 0.029221741763238503), ('4x4_714', 0.02909088741202722), ('4x4_115', 0.02864570258270748), ('4x4_15', 0.028550789385729407), ('4x4_594', 0.02834769086183192), ('4x4_964', 0.027997976143968588), ('4x4_383', 0.027833519665580644), ('4x4_210', 0.027657537279671306), ('4x4_453', 0.027409877510756544), ('4x4_456', 0.027379018116720485), ('4x4_444', 0.027309274309207435), ('4x4_698', 0.027146300142405135), ('4x4_439', 0.026930503252062653), ('4x4_225', 0.02684575990362647), ('4x4_238', 0.02674676339538344), ('4x4_308', 0.026722021570498092), ('4x4_610', 0.02654148341807042), ('4x4_839', 0.026280060621615888), ('4x4_829', 0.026207256372768307), ('4x4_187', 0.025933145884838454), ('4x4_140', 0.02576918946932752), ('4x4_139', 0.025744854070568462), ('4x4_356', 0.025387930720069025), ('4x4_709', 0.025159865754988874), ('4x4_64', 0.024934386321507206), ('4x4_4', 0.024793878088176204), ('4x4_553', 0.024752605892296376), ('4x4_671', 0.024607566524417486), ('4x4_324', 0.02450017403924397), ('4x4_51', 0.024488313146588465), ('4x4_752', 0.024391908747373785), ('4x4_406', 0.024360943736744772), ('4x4_551', 0.0243459773731381), ('4x4_908', 0.02430633229036155), ('4x4_247', 0.024290782259893288), ('4x4_413', 0.02383011545439827), ('4x4_694', 0.023730305093983866), ('4x4_598', 0.023708755890453227), ('4x4_329', 0.023673512689473576), ('4x4_593', 0.023671744785679887), ('4x4_862', 0.023668320428239537), ('4x4_360', 0.02349488827479241), ('4x4_81', 0.023465765561553085), ('4x4_114', 0.023183459759798762), ('4x4_979', 0.023106047105197074), ('4x4_794', 0.022902974382226535), ('4x4_278', 0.022900398657804077), ('4x4_589', 0.022891686139808187), ('4x4_801', 0.022586870544937673), ('4x4_960', 0.022409976334329586), ('4x4_905', 0.022399434599946852), ('4x4_522', 0.022391867551169023), ('4x4_759', 0.022306282249584943), ('4x4_669', 0.022239397355741552), ('4x4_242', 0.022076088868964845), ('4x4_473', 0.022001401613982847), ('4x4_741', 0.02200128642549011), ('4x4_481', 0.021972938424526897), ('4x4_565', 0.02195174889799497), ('4x4_38', 0.021888802754457853), ('4x4_162', 0.021717891067945964), ('4x4_350', 0.021697899955058882), ('4x4_799', 0.02157829642577185), ('4x4_907', 0.021144584263142092), ('4x4_557', 0.021111621455334625), ('4x4_28', 0.020867847669473444), ('4x4_240', 0.020837602067490472), ('4x4_175', 0.02080602411208821), ('4x4_853', 0.02036477587083876), ('4x4_667', 0.020353328941285855), ('4x4_12', 0.02023913858219144), ('4x4_564', 0.0202316696458191), ('4x4_975', 0.02020079090974647), ('4x4_385', 0.020134895545446688), ('4x4_338', 0.020087572119285472), ('4x4_130', 0.02007966845286997), ('4x4_638', 0.01998096872114243), ('4x4_630', 0.019594826670180865), ('4x4_300', 0.019331649915440008), ('4x4_251', 0.019134419745467), ('4x4_32', 0.018953301610349492), ('4x4_568', 0.018914378834021885), ('4x4_797', 0.018756006377002473), ('4x4_95', 0.018719843211585895), ('4x4_44', 0.01870268443648261), ('4x4_569', 0.018673016374808494), ('4x4_427', 0.018578139230051537), ('4x4_562', 0.018354015166064507), ('4x4_677', 0.0183387573600108), ('4x4_96', 0.01830558507203114), ('4x4_284', 0.01827491522543789), ('4x4_367', 0.01820999059920301), ('4x4_983', 0.01818522324658222), ('4x4_654', 0.018174478870058592), ('4x4_395', 0.01817437165145015), ('4x4_245', 0.018101795199483035), ('4x4_778', 0.0180186001406025), ('4x4_934', 0.01791932765497927), ('4x4_419', 0.017910897448242046), ('4x4_206', 0.017592265497073423), ('4x4_602', 0.017588104857647023), ('4x4_45', 0.017526670526354097), ('4x4_8', 0.01747788723679448), ('4x4_417', 0.017448724191995994), ('4x4_403', 0.017444713821998692), ('4x4_779', 0.017348926883200375), ('4x4_209', 0.017316789359820835), ('witness_8', 0.01728518478337534), ('4x4_488', 0.017274510358472932), ('4x4_327', 0.01718986156198112), ('4x4_591', 0.01686805182700552), ('4x4_380', 0.01684884323740488), ('4x4_845', 0.016602538081059607), ('4x4_747', 0.016537541110023285), ('4x4_644', 0.016495818893439748), ('4x4_877', 0.016355611849889283), ('3x3_662', 0.016311938263585683), ('4x4_724', 0.016307645241324584), ('4x4_237', 0.016257590464533803), ('4x4_857', 0.016091298280895047), ('4x4_430', 0.01596322771778687), ('4x4_927', 0.01594391596373521), ('4x4_281', 0.015818905038228314), ('4x4_71', 0.01577342878847279), ('4x4_993', 0.015742975595175124), ('4x4_916', 0.015717532829407185), ('4x4_904', 0.015554246884227364), ('4x4_931', 0.015553999125709234), ('4x4_220', 0.015501018489895232), ('4x4_498', 0.015359989984671047), ('4x4_179', 0.015174407265666646), ('4x4_999', 0.015119303591448573), ('4x4_464', 0.015106245861556317), ('4x4_549', 0.014770351919745807), ('4x4_43', 0.014721812768802193), ('4x4_168', 0.014677828623944204), ('4x4_725', 0.014640404604601844), ('4x4_391', 0.014609141315697622), ('4x4_334', 0.014586139215008684), ('4x4_511', 0.014571351702801471), ('4x4_984', 0.014425300391205318), ('4x4_945', 0.014408675297262419), ('4x4_935', 0.014328980138630813), ('4x4_763', 0.014274161615098485), ('4x4_929', 0.014236360725342312), ('4x4_297', 0.014040665336751346), ('4x4_876', 0.01401975333118987), ('4x4_494', 0.013943779310781298), ('4x4_449', 0.013919332979882909), ('4x4_701', 0.013914802259468121), ('4x4_974', 0.013732867332709184), ('4x4_409', 0.013711039471891784), ('4x4_921', 0.013662766296927453), ('4x4_450', 0.013594982920321961), ('4x4_275', 0.013550572828506307), ('4x4_193', 0.013533940026995366), ('4x4_648', 0.01350355665198758), ('4x4_889', 0.013282400577696836), ('4x4_93', 0.013242309984875302), ('4x4_116', 0.013215182657430418), ('4x4_529', 0.013201682202774206), ('4x4_780', 0.013035783591969937), ('4x4_56', 0.013023468408066993), ('4x4_760', 0.01299530353273254), ('4x4_992', 0.012990866110579863), ('4x4_143', 0.012960982470862269), ('4x4_874', 0.01291519526774829), ('4x4_261', 0.012888979294397638), ('4x4_503', 0.012693269484165853), ('4x4_386', 0.012570747131975398), ('4x4_911', 0.012474136388687216), ('4x4_860', 0.012458096285344602), ('4x4_580', 0.012415424862139925), ('4x4_294', 0.012369869524110606), ('4x4_246', 0.012323622013256657), ('4x4_190', 0.012271694790129852), ('4x4_847', 0.012270356208271637), ('4x4_803', 0.012148739065876189), ('4x4_811', 0.012123454580137522), ('4x4_852', 0.012109579248870685), ('4x4_851', 0.012101255005314344), ('4x4_764', 0.012088317911050181), ('4x4_111', 0.012070117300197171), ('4x4_743', 0.012014022374757643), ('4x4_402', 0.012005722288449589), ('4x4_631', 0.011935580344998436), ('4x4_753', 0.011720885117231655), ('4x4_796', 0.011558796699054295), ('4x4_740', 0.011466591613067699), ('4x4_989', 0.011435818795447182), ('4x4_471', 0.011396025498594161), ('4x4_838', 0.011293306761618252), ('4x4_332', 0.011267564556959975), ('4x4_955', 0.011253607861995023), ('4x4_85', 0.011247576956204762), ('4x4_988', 0.011136066027136843), ('4x4_504', 0.011103289349450983), ('4x4_536', 0.01109917406578287), ('4x4_769', 0.01106506493568404), ('4x4_105', 0.011052094596496254), ('4x4_423', 0.011007379511340168), ('4x4_215', 0.010862937283854596), ('4x4_231', 0.010848070668018043), ('4x4_479', 0.010682118395498723), ('4x4_643', 0.010664279623358187), ('4x4_613', 0.010619898895065007), ('4x4_357', 0.01058963594891648), ('4x4_253', 0.010273769954793378), ('4x4_394', 0.010210530972493593), ('4x4_232', 0.010130648734153237), ('4x4_641', 0.010119054832335451), ('4x4_539', 0.010103285371450473), ('4x4_22', 0.010084621723757802), ('4x4_835', 0.009922655566466008), ('4x4_248', 0.00977724810715624), ('4x4_637', 0.009614008671492796), ('4x4_575', 0.009426153333819095), ('4x4_198', 0.009348550463775288), ('4x4_200', 0.009324980026021152), ('4x4_155', 0.00917387019193411), ('4x4_344', 0.009029033956060651), ('4x4_521', 0.008889730208464734), ('4x4_77', 0.00888722210424967), ('4x4_773', 0.008800390675697786), ('4x4_751', 0.008787301175034911), ('4x4_2', 0.008756881995194695), ('4x4_148', 0.008746249527356623), ('4x4_34', 0.00872562283641205), ('4x4_326', 0.008699592346549009), ('4x4_533', 0.008634206422760847), ('4x4_640', 0.008633909868479433), ('4x4_746', 0.008545401926850736), ('4x4_107', 0.008242524303927658), ('4x4_817', 0.00814170080701355), ('4x4_590', 0.008122857721644661), ('4x4_320', 0.0079090223246106), ('4x4_863', 0.007819528288456976), ('4x4_915', 0.00758436365560358), ('4x4_682', 0.007475209839984964), ('4x4_400', 0.007341492939806243), ('4x4_679', 0.007196196578461838), ('4x4_537', 0.007161830593052426), ('4x4_482', 0.007154971494152873), ('4x4_270', 0.007141791833160797), ('4x4_207', 0.007136277337280583), ('4x4_374', 0.00711331940765237), ('4x4_287', 0.007096891785557458), ('4x4_966', 0.0070801896414807225), ('4x4_733', 0.007002576573414755), ('4x4_922', 0.006994510070203418), ('4x4_750', 0.006935980223396644), ('4x4_465', 0.006824134633417087), ('4x4_317', 0.0068013720955010645), ('4x4_617', 0.006721277527394571), ('4x4_879', 0.006680732711582597), ('4x4_666', 0.006657134990158052), ('4x4_262', 0.0066387178237325645), ('4x4_720', 0.006390972149334166), ('4x4_492', 0.006375734205733891), ('4x4_425', 0.006297726828330204), ('4x4_791', 0.006190127242988529), ('3x3_290', 0.006186045029859337), ('4x4_880', 0.006176380108835625), ('4x4_432', 0.0061464413159921785), ('4x4_692', 0.005956226457606171), ('4x4_35', 0.0059398653414338256), ('4x4_98', 0.00593358812905336), ('4x4_352', 0.0058864167048735165), ('4x4_672', 0.0058272682975818975), ('4x4_761', 0.005822657879515099), ('4x4_491', 0.0057984254177382176), ('4x4_354', 0.005706664324332352), ('4x4_194', 0.005705760421961473), ('4x4_131', 0.005654934712449276), ('4x4_373', 0.005651668623282379), ('4x4_820', 0.005627412394668976), ('4x4_910', 0.005535123418055697), ('4x4_826', 0.005381011366132661), ('4x4_861', 0.005154059810134596), ('4x4_956', 0.005123975274939833), ('4x4_525', 0.005032661349626539), ('4x4_636', 0.00469113537879056), ('4x4_381', 0.004668468844845362), ('4x4_176', 0.004525770586377759), ('4x4_578', 0.004525747516606478), ('4x4_387', 0.004474077757530029), ('4x4_642', 0.004351159086027601), ('4x4_306', 0.004275585150253702), ('4x4_24', 0.0042406710502529915), ('4x4_359', 0.004182796950159766), ('4x4_268', 0.004182492231641602), ('4x4_675', 0.004045493208738092), ('4x4_579', 0.003991981868203766), ('4x4_112', 0.003991571674329931), ('4x4_461', 0.003935350025986308), ('4x4_399', 0.003927280491607192), ('4x4_337', 0.0038607488175245305), ('4x4_550', 0.0036582366403735906), ('4x4_687', 0.0035999585970176284), ('4x4_161', 0.003428861211753953), ('4x4_603', 0.00339568306312178), ('4x4_681', 0.0033486974017370385), ('4x4_440', 0.003186185573874446), ('4x4_25', 0.003025301671993013), ('4x4_828', 0.0029701748807535477), ('4x4_620', 0.0028304370867371256), ('4x4_962', 0.0024101419582266735), ('4x4_84', 0.0023881265409410645), ('4x4_807', 0.002312049250387364), ('4x4_18', 0.0022812706112211897), ('4x4_821', 0.0022276821335969962), ('4x4_684', 0.0019288240545611063), ('4x4_134', 0.0015911654447924007), ('4x4_46', 0.0014705279185001723), ('4x4_363', 0.0011197594930368573), ('4x4_446', 0.001027254156869647), ('4x4_976', 0.0007959622558211785), ('4x4_822', 0.000746148976998357), ('4x4_104', 0.0005907210530073007), ('4x4_885', 0.0003439549679873587), ('4x4_581', 0.00030330220117579453), ('4x4_881', 2.1850234202334978e-05), ('4x4_526', -1.612411022976452e-05), ('4x4_11', -2.3111324111575993e-05), ('4x4_216', -0.0003025866595634115), ('4x4_152', -0.0006595878130435198), ('4x4_690', -0.000752713081557193), ('4x4_472', -0.0011979752649832589), ('4x4_512', -0.0012955263112879646), ('4x4_606', -0.0013425417563635276), ('4x4_623', -0.0014604019041390996), ('4x4_882', -0.001571957577425447), ('4x4_563', -0.0015954826435621292), ('4x4_442', -0.0017189586844678195), ('4x4_618', -0.0017682023021857137), ('4x4_138', -0.001777622669066091), ('4x4_782', -0.001796747327733601), ('4x4_313', -0.001823985972334627), ('4x4_204', -0.001951535413458094), ('4x4_263', -0.0019737981622605376), ('4x4_133', -0.0020459211065295777), ('4x4_674', -0.002218796473873451), ('4x4_441', -0.002326528815181932), ('4x4_260', -0.0024109520024919676), ('4x4_715', -0.0024359996885347976), ('4x4_299', -0.0029623261036399932), ('4x4_227', -0.0030386243280648723), ('4x4_355', -0.003219715585891734), ('4x4_685', -0.0035251575180483877), ('4x4_347', -0.0035263641707896846), ('4x4_887', -0.0036450547334339425), ('4x4_968', -0.0036666942609826516), ('4x4_998', -0.003788266245985786), ('4x4_490', -0.003845747426152495), ('4x4_328', -0.003899794206999028), ('4x4_928', -0.004441486997906312), ('4x4_903', -0.004452956004959196), ('4x4_869', -0.00479337782671876), ('4x4_708', -0.004803024446024764), ('4x4_119', -0.005132517693854544), ('4x4_516', -0.0052259408300619654), ('4x4_588', -0.005606951055327757), ('4x4_372', -0.0057772260973982656), ('4x4_584', -0.005920676626560115), ('4x4_228', -0.006017153277102677), ('4x4_73', -0.006526486193366274), ('4x4_312', -0.0067883752273513185), ('4x4_969', -0.007052633234100065), ('4x4_377', -0.007268680998626508), ('4x4_854', -0.008311755256822794), ('4x4_154', -0.008588694053164836), ('4x4_609', -0.009234828921887728), ('4x4_996', -0.009873985602714692), ('4x4_27', -0.010015513521725418), ('4x4_582', -0.010096228789036661), ('4x4_830', -0.01100146301718832), ('4x4_924', -0.011744836715523635), ('4x4_997', -0.011872891383237928), ('4x4_925', -0.01302206390844869), ('4x4_547', -0.013685390018911711), ('4x4_127', -0.014068456642429578), ('4x4_572', -0.015713467763025304), ('4x4_570', -0.015904600689032256), ('4x4_6', -0.016896426010508916), ('4x4_774', -0.017889301303395562), ('4x4_768', -0.020297566331943)]
argmax_p_DotProds = ('4x4_283', 1.3617835064248265)
argmax_p_Cosines = ('4x4_75', 0.053513762030853555)

witness_puzzle = WitnessPuzzle ()
d = {}
for file in os.listdir('puzzles_4x4/'):
    print("")
    print("file", file)
    full_filename = 'puzzles_4x4/' + file
    if "Idxs" in file or file == "BFS_memory_4x4.pkl" or ".py" in file:
        continue

    some_dict = {}  # TODO: debug
    if "Rank_" in file:
        full_filename = 'puzzles_4x4/' + file
        object = open_pickle_file(full_filename)
        print ("rank filename", file)
        if "DotProd" in file:
            print("file", file)
            object[0][1] = Rank_DotProds
            total = 0
            for i, sublist in enumerate(object[0]):
                sublist_dict = {"1x2": 0, "1x3": 0, "2x2": 0, "3x3": 0, "4x4": 0, "w": []}
                L = len(sublist)
                print("len(sublist) =", L)
                total += L
                # TODO:
                for tup_item in sublist:
                    name = tup_item[0]
                    if "1x2" in name:
                        sublist_dict["1x2"] += 1
                    elif "1x3" in name:
                        sublist_dict["1x3"] += 1
                    elif "2x2" in name:
                        sublist_dict["2x2"] += 1
                    elif "3x3" in name:
                        sublist_dict["3x3"] += 1
                    elif "4x4" in name:
                        sublist_dict["4x4"] += 1
                    else:
                        new_name = map_witness_puzzles_to_dims(name)
                        sublist_dict[new_name] += 1
                        sublist_dict['w'] += [name]
                some_dict["sublist_" + str(i)] = sublist_dict
            print("some_dict =", some_dict)
            print("")
            print("total", total)
            print("")

            # assert False
    #
    #     elif "Cosine" in file:
    #         print("file", file)
    #         object[0][1] = Rank_Cosines
    #         for sublist in object[0]:
    #             print("len(sublist) =", len(sublist))
    #
    #     flat = flatten_list(flatten_list(object))
    #     assert len (flat) == 2369
    #     list_names, list_vals = separate_names_and_vals(flat)
    #
    #      d = get_witness_ordering (flat, d)
    #     # special_x, special_y = find_special_vals (object)  # we don't use this at all
    #     # for i, sub_sublist in enumerate(object[0]):
    #     #     print("len(sub_sublist) =", len(sub_sublist))
    #     #     for tuple_el in sub_sublist:
    #     #         p_name = tuple_el[0]
    #     #         if "wit" in p_name:
    #     #             print("pname ", p_name, " found in sublist ", i)
    #     print("")
    #     # continue  # TODO: debug -- uncomment
    #
    #     if "DotProd" in file:
    #         title_name = "grad_c(p) * (theta_{t+1} - theta_t)"
    #         # sublist = d[file]
    #     elif "Cosine" in file:
    #         title_name = "cosine(angle(grad_c(p), (theta_{t+1} - theta_t)))"
    #     else:
    #         title_name = "Levin Cost"
    #
    #     plots_filename = os.path.join(plots_path, "Zoomed_" + file.split("_BFS_4x4.pkl")[0])
    #     print("plots_filename", plots_filename)
    #
    #     special_x = None
    #     special_y = None
    #     plot_data(list_vals, idx_object[0], title_name, plots_filename, special_x, special_y)
    # # continue # TODO: debug -- uncomment
    #
    # elif "Ordering" in file:
    #     object = open_pickle_file('puzzles_4x4/' + file)
    #     if "DorProds" in file:
    #         object[0][1] = argmax_p_DotProds
    #     elif "Cosines" in file:
    #         object[0][1] = argmax_p_Cosines
    #     # print("len(ordering object)", len(object[0]))
    #     print ("ordering filename", file)
    #     print(object)
    #     print("")
    #
    #     # continue
    #     filename = file.split("_BFS")[0]
    #     print ("filename", filename)
    #     imgs_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), puzzles_path)
    #     print ("specific images path =", imgs_path)
    #     if not os.path.exists (imgs_path):
    #         os.makedirs (imgs_path, exist_ok=True)
    #
    #     for tup in object[0]:
    #         p_name = tup[0]
    #         witness_puzzle.read_state ("../problems/witness/puzzles_4x4/" + p_name)
    #         img_file = os.path.join(imgs_path, p_name + ".png")
    #         print("img file", img_file)
    #         witness_puzzle.save_figure (img_file)

# print("")
# print("d", d)
# print("")