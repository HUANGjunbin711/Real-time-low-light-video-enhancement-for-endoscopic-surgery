import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login(authToken="")
files = synapseutils.syncFromSynapse(syn, ' syn18779624 ')