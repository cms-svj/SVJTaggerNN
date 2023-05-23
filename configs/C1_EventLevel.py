from magiconfig import MagiConfig

config = MagiConfig(dataset=MagiConfig(), features=MagiConfig(), training=MagiConfig(), hyper=MagiConfig())
config.dataset.path = "~/uscmshome/nobackup/SVJ/eventLevelTagger/trainingFiles/"
config.dataset.signal = {
							"signal":		[
												"tree_SVJ_mMed-1000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-1500_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-100_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-1_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-20_rinv-0p1_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-20_rinv-0p5_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-20_rinv-0p7_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-2000_mDark-50_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-3000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-4000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-600_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
												"tree_SVJ_mMed-800_mDark-20_rinv-0p3_alpha-peak_yukawa-1_NN",
											],
}
config.dataset.background =  {	
								"QCD": 		[
												"tree_QCD_Pt_300to470_NN",
												"tree_QCD_Pt_470to600_NN",
												"tree_QCD_Pt_600to800_NN",
												"tree_QCD_Pt_800to1000_NN",
												"tree_QCD_Pt_1000to1400_NN",
												"tree_QCD_Pt_1400to1800_NN",
												"tree_QCD_Pt_1800to2400_NN",
												"tree_QCD_Pt_2400to3200_NN"
											],
								"TTJets": 	[
												"tree_TTJets_DiLept_genMET-150_NN",
												"tree_TTJets_DiLept_NN",
												"tree_TTJets_HT-1200to2500_NN",
												"tree_TTJets_HT-2500toInf_NN",
												"tree_TTJets_HT-600to800_NN",
												"tree_TTJets_HT-800to1200_NN",
												"tree_TTJets_Incl_NN",
												"tree_TTJets_SingleLeptFromTbar_genMET-150_NN",
												"tree_TTJets_SingleLeptFromTbar_NN",
												"tree_TTJets_SingleLeptFromT_genMET-150_NN",
												"tree_TTJets_SingleLeptFromT_NN"
											]
}
config.dataset.sample_fractions = [0.7, 0.15, 0.15]
config.features.eventVariables = [
#'njets', 
'njetsAK8', 
#'nb', 
#'nl', 
#'nnim', 
#'ht', 
#'st', 
# 'met', 
'mT', 
'METrHT_pt30', 
#'METrST_pt30', 
'dEtaj12AK8', 
'dRJ12AK8', 
'dPhiMinjMETAK8', 
#'dPhij1rdPhij2AK8', 
]
#config.features.jetConst = None
config.features.jetVariables = [
#'fjw',
'jPtAK8', 
'jEtaAK8', 
'jPhiAK8', 
# 'jGirthAK8',
# 'jAxismajorAK8',
# 'jAxisminorAK8',
# 'jTau1AK8',
# 'jTau2AK8',
# 'jTau3AK8',
# 'jSoftDropMassAK8', 
'dPhijMETAK8', 
#'JetsAK8_hvCategory', 
#'JetsAK8_darkPtFrac', 
# 'nnOutput',
]
config.features.numOfJetsToKeep = 2
# hyperparameters
config.hyper.learning_rate = 0.001
config.hyper.batchSize = 512
config.hyper.num_of_layers = 5
config.hyper.num_of_nodes = 200
config.hyper.dropout = 0.3
config.hyper.epochs = 80
config.hyper.lambdaTag = 1.0
config.hyper.lambdaReg = 0.0
config.hyper.lambdaGR = 1.0 # keep this at 1 and change lambdaReg only
config.hyper.lambdaDC = 0.0
config.hyper.rseed = 100
config.hyper.num_classes = 3
