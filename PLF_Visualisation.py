# Imports
def importDependencies():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpltlib
    import math
    import os

    # Return all the imports as a tuple
    return pd, np, plt, mcolors, LinearSegmentedColormap, mpltlib, math, os

pd, np, plt, mcolors, LinearSegmentedColormap, mpltlib, math, os = importDependencies()

# Add standard deviation columns
## Notes:
# Strings "FZ1_F (ms-status_RAT1)" refer to a sample group within one of the datasets. As different datasets have different column
# names these can be changed depending on the column names applicable to a MPLF results dataset.
# 2 Sets of standard deviations are applied as as there are 2 sample groups; in this case "Old" and "Young" rat specimens. 
# If MPLF results.tsv export column indexes are always the same, the names of PSM sample columns could be replaced with column indexes. 
def addStandardDeviation(data, sampleNames):
    compGroup1Names = sampleNames[0]
    compGroup2Names = sampleNames[1]

    data['SD_g1'] = 0
    data['SD_g2'] = 0

    for i, row in data.iterrows():
        data.at[i, 'SD_g1'] = row[compGroup1Names].std()
        data.at[i, 'SD_g2'] = row[compGroup2Names].std()
    return data

# Filter for significants - returns an object with a list 'prots' of unique proteins that are significant and the data with only significant proteins included 'data'
## The filter ensures there is at least an average peptide spectral count (PSC) of > 1 in both conditions (old vs young)
## The difference in average peptide spectral count must also be greater than 1 between the conditions
## The p-value must be less than 0.05 level of significance
def filterData(data, g1AverageColName, g2AverageColName, pValueColName, *args, **kwargs):
    Thresh = kwargs.get('thresh', 0.05)
    
    true_sigs = data[data[g1AverageColName] > 0]
    true_sigs = true_sigs[true_sigs[g2AverageColName] > 0]
    true_sigs = true_sigs[abs((true_sigs[g2AverageColName] - true_sigs[g1AverageColName])) > 1]
    
    if pValueColName in data.columns:
        only_sig = true_sigs[true_sigs[pValueColName] < Thresh]
    
    uniq = only_sig['Uniprot_ID'].unique()

    res = {
        'data': data[data['Uniprot_ID'].isin(uniq)],
        'prots': uniq
    }
    return res

## Finds proteins unique to each dataset and the ones found in both datasets
# Find common proteins - takes in two lists
def findCommon(data):
    sets = []
    # Convert lists to sets
    for i in data:
        if type(i) == pd.core.frame.DataFrame:
            dataset = set(i['Uniprot_ID'])
            sets.append(dataset)
        else:
            dataset = set(i)
            sets.append(dataset)

    # Find intersection
    common_strings = set.intersection(*sets)

    res = {}

    # Find unique proteins for each set and store in the dictionary
    for i, s in enumerate(sets):
        # Union of all sets except the current one
        other_sets_union = set.union(*(sets[:i] + sets[i+1:]))
        # Unique proteins in the current set
        unique_to_current = s - other_sets_union
        # Add the unique proteins to the dictionary
        res[f"set_{i+1}"] = unique_to_current

    # Convert the result back to a list if needed
    common_strings_list = list(common_strings)
    proteins_in_both = common_strings_list
    res['common_to_all'] = common_strings

    return res

## Finds proteins unique to each dataset and the ones found in both datasets
# Find common proteins - takes in two lists
def findCommonProteinsAndDomains(data):
    sets = []
    # Convert lists to sets
    for i in data:
        if type(i) == pd.core.frame.DataFrame:
            dataset = set(i['Uniprot_ID'])
            sets.append(dataset)
        else:
            dataset = set(i)
            sets.append(dataset)

    # Find intersection
    common_strings = set.intersection(*sets)

    res = {}

    # Find unique proteins for each set and store in the dictionary
    for i, s in enumerate(sets):
        # Union of all sets except the current one
        other_sets_union = set.union(*(sets[:i] + sets[i+1:]))
        # Unique proteins in the current set
        unique_to_current = s - other_sets_union
        # Add the unique proteins to the dictionary
        res[f"set_{i+1}"] = unique_to_current

    # Convert the result back to a list if needed
    common_strings_list = list(common_strings)
    proteins_in_both = common_strings_list
    res['common_to_all'] = common_strings

    return res

# Defines colour palette to be used for the heatmaps 
colors = [(1, 1, 1), (1, 0, 0)]  # Transition from white (1, 1, 1) to red (1, 0, 0)
n_bins = 100  # Number of discrete steps in the colormap
cmap_name = 'white_to_red'  # Name of the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


# Visualise average peptide spectral counts
# Builds Bargraphs with heatmaps:
# requires: d = data (dataframe) for a particular protein or entire dataset of all proteins if protein specified,
#   diffColName = name of the column with average normalised difference in peptide spectral count (string),
#   g = name of the group that average peptide spectral count column that you wish to look at (string),
#   g1AverageColName = name of the group 1 average peptide spectral count column (string),
#   g2AverageColName = name of the group 2 average peptide spectral count column (string),
#   pValueColName = name of the column containing p_value (string),
# Optional: save = do you want the plot to be saved as an svg (Boolean),
#   t = extension for main title string (String)
#   p = protein_id (string)
def buildGraph(d, g, g1AverageColName, g2AverageColName, pValueColName, *args, **kwargs):
    # Save variables
    save = kwargs.get('save', None)
    
    protein = kwargs.get('p', None) 

    t = kwargs.get('t', '')

    directory = kwargs.get('directory', None)

    if protein:
        proteins = []
        listProteins = protein.split()
        for prot in listProteins:
            if d['Uniprot_ID'].str.contains(prot, na=False).any():
                proteins.append(prot)
    else:
        proteins = d['Uniprot_ID'].unique()

    for p in proteins:
    
        Protein = d[d['Uniprot_ID'] == p]
        SegmentValue = 50
        BarPos = SegmentValue / 2
        pml = Protein['Domain_Finish'].max()
        previous_value = 0

        GraphData = {
            'Segments': [],
            'Values': [],
            'HeatValues': [],
            'SD': [],
            'p_values': [],
            'ticks': [0]
        }

        # Process data for plotting
        for index, row in Protein.iterrows():
            domainFinish = row['Domain_Finish'] - ((row['Domain_Finish'] - previous_value)/2)
            Aver = row[g]
            GraphData['Segments'].append(domainFinish)
            GraphData['Values'].append(Aver)
            previous_value = row['Domain_Finish']
            if g == g1AverageColName:
                sdg = 'SD_g1'
                GraphData['SD'].append(row[sdg])
            else: 
                sdg = 'SD_g2'
                GraphData['SD'].append(row[sdg])
            # Standard Deviation
            #p-values
            GraphData['p_values'].append(row[pValueColName])
            
            GraphData['ticks'].append(row['Domain_Finish'])
        
        GraphData['ticks'].append(pml)

        GraphData['HeatValues'] = np.array(GraphData['Values'])  # This should be outside the loop

        fig, axs = plt.subplots(2, 1, figsize=(15, 7.5), gridspec_kw={'height_ratios': [2, 0.15]}, layout='constrained')

        # Barchart
        ax1 = axs[0]
        ax2 = axs[1]

        ticks = GraphData['ticks']

        # highest value of for y axis combine value plus standard deviation
        max_y_with_SD = [a + b for a, b in zip(GraphData['Values'], GraphData['SD'])]
        
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([])
        ax1.set_yticks(np.arange(np.floor(min(GraphData['Values'])), np.ceil(max(max_y_with_SD)) + 1, 1))
        ax1.set_yticklabels(np.arange(np.floor(min(GraphData['Values'])), np.ceil(max(max_y_with_SD)) + 1, 1), fontsize=8)
        ax1.set_xlim(0, pml)
        if  max(GraphData['SD']) == 0:
            ax1.set_ylim(0, max(GraphData['Values']) + 0.25)
        else: 
            ax1.set_ylim(0, max(GraphData['Values']) + max(GraphData['SD']) + 0.25)
        ax1.bar(GraphData['Segments'], GraphData['Values'], yerr = GraphData['SD'], width=7) 
        if t:
            ax1.set_title("{} {}".format(p, t), pad=20)
        else:
            ax1.set_title("{} average peptide spectral counts in {}".format(p, g), pad=20)
        ax1.set_ylabel('Average PSC')

        # Heatmap
        ax2.set_xticks(ticks)
        ax2.set_xlim(0, pml)
        ax2.set_xticklabels(ticks, fontsize=8)
        ax2.set_ylim(0, 0.1)
        ax2.set_yticks([])
        ax2.set_xlabel('peptide position')
        
        ax1.tick_params(axis='x', labelrotation=45)  # Rotate labels by 45 degrees
        ax2.tick_params(axis='x', labelrotation=45)  # Rotate labels by 45 degrees
        
        ax2.text(pml, -0.15, f'{pml}',
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=8,  # Adjust font size as needed
        bbox=dict(facecolor='white', alpha=0.5)  # Optional: Add a background to the label
        )
        
        ax1.axhline(y=np.ceil(max(GraphData['Values'])), color='red', linestyle='--')
    
        # Now plot the heatmap with pcolormesh or imshow
        ax2.pcolormesh(ticks[:-1], [0, 1], [GraphData['HeatValues']], cmap=cm, shading='flat')

        plt.tight_layout()
        if save:
            if not os.path.exists(directory):
                print(f'directory = {directory}, directory made')
                os.makedirs(directory)
            plt.savefig('{}\Bar_plot_{}_{}.svg'.format(directory, p, g), format='svg')
        plt.close()


# Builds difference in average spectral counts line/scatter plot:
# requires: 
#   d = data (dataframe) for a particular protein or entire dataset of all proteins if protein specified,
#   diffColName = name of the column with average normalised difference in peptide spectral count (string),
#   pValueColName = name of the column containing p_value (string),
#   g1AverageColName = name of the group 1 average peptide spectral count column (string),
#   g2AverageColName = name of the group 2 average peptide spectral count column (string)
# optional: 
#   p = uniprot_id (string), 
#   t = graph title (string),
#   xlab = x axis label (string),
#   ylab = y axis label (string),
#   g1lab = comparison group 1 label for legend (string)
#   g2lab = comparison group 2 label for legend (string)
#   save = whether you wish to save graph to local directory (bool)
#   addRegion = [(regionlabel, peptideposition, labelposition, labeloffset)], (list of tuples), eg: ['NC1-start', 1208, 'center', 0.01)] labeloffset=0.01 (recommended)

def buildDiffGraph(d, diffColName, pValueColName, g1AverageColName, g2AverageColName, *args, **kwargs):

    protein = kwargs.get('p', None)
    t = kwargs.get('t', '')
    xlab = kwargs.get('xlab', None)
    ylab = kwargs.get('ylab', None)
    g1lab = kwargs.get('g1lab', None)
    g2lab = kwargs.get('g2lab', None)
    addRegion = kwargs.get('addRegion', None)
    save = kwargs.get('save', None)
    directory = kwargs.get('directory', '')
    Thresh = kwargs.get('thresh', 0.05)

    if protein:
        proteins = []
        listProteins = protein.split()
        for prot in listProteins:
            if d['Uniprot_ID'].str.contains(prot, na=False).any():
                proteins.append(prot)
    else:
        proteins = d['Uniprot_ID'].unique()

    for p in proteins:
    
        Protein = d[d['Uniprot_ID'] == p]
        SegmentValue = 50
        BarPos = SegmentValue / 2
        pml = Protein['Domain_Finish'].max()
        previous_value = 0
        
        GraphData = {
            'Segments': [],
            'Values': [],
            'p_values': [],
            'ticks': []
        }

        # Process data for plotting
        for index, row in Protein.iterrows():
            domainFinish = row['Domain_Finish'] - ((row['Domain_Finish'] - previous_value)/2)
            Aver = row[diffColName]
            GraphData['Segments'].append(domainFinish) 
            GraphData['Values'].append(Aver)
            GraphData['HeatValues'] = np.array(GraphData['Values'])
            GraphData['p_values'].append(row[pValueColName])
            previous_value = row['Domain_Finish'] 
            GraphData['ticks'].append(row['Domain_Finish'])
        
        GraphData['ticks'].append(pml)

        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(1, 1, 1 )
        if t:
            ax.set_title("{} {}".format(p, t), pad=30)
        else:
            ax.set_title("{} Diff group 1 vs group 2 (average group 1 - group 2 / 50)".format(p), pad=30)
        if xlab: 
            ax.set_ylabel('{}'.format(xlab), labelpad=15)
        else: 
            ax.set_ylabel('(average group 1 - group 2 / 50)', labelpad=15)
        if ylab:
            ax.set_xlabel('{}'.format(ylab), labelpad=15)
        else:
            ax.set_xlabel('peptide position', labelpad=15)

        ticks = GraphData['ticks']

        ax.set_xticks(ticks)
        ax.set_xlim(0, pml)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.tick_params(axis='x', labelrotation=45)  # Rotate labels by 45 degrees
        ax.plot(GraphData['Segments'], GraphData['Values'])
        ax.scatter(GraphData['Segments'], GraphData['Values'])    
        ax.axhline(y=0, color='black', linestyle='--')
        # Calculate the maximum absolute value of y for setting limits
        y_max = max(abs(min(GraphData['Values'])), abs(max(GraphData['Values'])))
        plt.ylim(-y_max, y_max)
        
        # Add regions labels
        if addRegion:
            for specRegion in addRegion:
                # print(specRegion)
                plt.axvline(x=specRegion[1], color='r', linestyle='--', label='{} at x={}'.format(specRegion[0], specRegion[1]))
                plt.text(specRegion[1], y_max +  specRegion[3], '{} ({})'.format(specRegion[0], specRegion[1]), color='r', ha= specRegion[2])

        for i, p_value in enumerate(GraphData['p_values']):
            region = Protein.iloc[i]
            if p_value < 0.001 and abs(region[g2AverageColName] - region[g1AverageColName]) > 1 and region[g2AverageColName] > 0 and region[g1AverageColName] > 0:
                annotation = '***'
            elif p_value < 0.01 and abs(region[g2AverageColName] - region[g1AverageColName]) > 1 and region[g2AverageColName] > 0 and region[g1AverageColName] > 0:
                annotation = '**'
            elif p_value < 0.05 and abs(region[g2AverageColName] - region[g1AverageColName]) > 1 and region[g2AverageColName] > 0 and region[g1AverageColName] > 0:
                annotation = '*'
            else:
                annotation = ''
            if annotation:
                plt.annotate(annotation, 
                             (GraphData['Segments'][i], GraphData['Values'][i]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        plt.tight_layout()
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f'directory = {directory}, directory made')
            plt.savefig('{}\Diff_{}.svg'.format(directory, p), format='svg')
    plt.close()


# Visaulise difference in spectral counts for 2 datasets
# Data type = list of data frames [data_1, data_2]
# requires: 
#   d = data (dataframe) for a particular protein or entire dataset of all proteins if protein specified,
#   diffColName = name of the column with average normalised difference in peptide spectral count (string),
#   pValueColName = name of the column containing p_value (string),
#   g1AverageColName = name of the group 1 average peptide spectral count column (string),
#   g2AverageColName = name of the group 2 average peptide spectral count column (string)
# optional:  
#   p = uniprot_id (string), 
#   t = graph title (string),
#   xlab = x axis label (string),
#   ylab = y axis label (string),
#   g1lab = comparison group 1 label for legend (string)
#   g2lab = comparison group 2 label for legend (string)
#   save = whether you wish to save graph to local directory (bool)
#   addRegion = [(regionlabel, peptideposition, labelposition, labeloffset)], (list of tuples), eg: ['NC1-start', 1208, 'center', 0.01)] labeloffset=0.01 (recommended)

def visualiseDiffComparisons(d, diffColName, pValueColName, g1AverageColName, g2AverageColName, *args, **kwargs):
    protein = kwargs.get('p', None)
    t = kwargs.get('t', '')
    xlab = kwargs.get('xlab', None)
    ylab = kwargs.get('ylab', None)
    g1lab = kwargs.get('g1lab', None)
    g2lab = kwargs.get('g2lab', None)
    addRegion = kwargs.get('addRegion', None)
    save = kwargs.get('save', None)
    directory = kwargs.get('directory', None)
    Thresh = kwargs.get('thresh', 0.05)
    
    if protein:
        data_1, data_2 = d
        data_1 = data_1[data_1['Uniprot_ID'] == protein]
        data_2 = data_2[data_2['Uniprot_ID'] == protein]
    else:
        data_1, data_2 = d
    
    # Check which proteins are present in both datasets and then create dataframes with only those proteins in them
    set1 = set(data_1['Uniprot_ID'])
    set2 = set(data_2['Uniprot_ID'])
    common_prots = set1.intersection(set2)
    common_prots = list(common_prots) # list of common proteins
    data_1 = data_1[data_1['Uniprot_ID'].isin(common_prots)]
    data_2 = data_2[data_2['Uniprot_ID'].isin(common_prots)]
    data = [data_1, data_2]
    
    # Build the difference graph 
    # Set up variables needed for axis creation
    # Start loop through list of proteins so each protein gets it's own graph
    for p in common_prots:
        
        # Initiate Graph data object
        GraphData = {}
        
        for i, d in enumerate(data):
            Protein = d[d['Uniprot_ID'] == p]  
            SegmentValue = 50
            BarPos = SegmentValue / 2
            pml = Protein['Domain_Finish'].max()
            previous_value = 0
            
            # Initiate object that will store the Graph plotting data - re-initiates every for every protein/graph
            data_name = f'data_{i + 1}'
            GraphData[data_name] = {
                'Segments': [],
                'Values': [],
                'AVG_g1': [],
                'AVG_g2': [],
                'p_values': [],
                'ticks': []
            }
            
            # Process data for plotting
            for index, row in Protein.iterrows():
                # Set the values for data_1 and data_2 
                domainFinish = row['Domain_Finish'] - ((row['Domain_Finish'] - previous_value)/2)
                Aver = row[diffColName]
                Avg_y = row[g1AverageColName]
                Avg_o = row[g2AverageColName]
                GraphData[data_name]['Segments'].append(domainFinish)
                GraphData[data_name]['Values'].append(Aver)
                GraphData[data_name]['AVG_g1'].append(Avg_y)
                GraphData[data_name]['AVG_g2'].append(Avg_o)
                GraphData[data_name]['p_values'].append(row[pValueColName])
                previous_value = row['Domain_Finish'] 
                GraphData[data_name]['ticks'].append(row['Domain_Finish'])
        
        GraphData[data_name]['ticks'].append(pml)
        
        # Build figure and plot data 
        # set up graph structure
        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(1, 1, 1 )
        if t:
            ax.set_title("{} {}".format(p, t), pad=30)
        else: 
            ax.set_title("{} average PSC difference between group 1 and group 2 (average group 1 - group 2 / 50)".format(p), pad=30)
        if ylab:
            ax.set_ylabel('{}'.format(ylab), labelpad=15)
        else:
            ax.set_ylabel('(average group 1 - group 2 / 50)', labelpad=15)
        if xlab:
            ax.set_xlabel('{}'.format(xlab), labelpad=15)
        else:
            ax.set_xlabel('peptide position', labelpad=15)
        ticks = GraphData[data_name]['ticks']

        # ticks = [0]
        # ticks = np.append(ticks, np.arange(0, pml, 50))
        # ticks = np.append(ticks, pml)

        ax.set_xticks(ticks)
        ax.set_xlim(0, pml)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.tick_params(axis='x', labelrotation=45)  # Rotate labels by 45 degrees

        # Plot actual data
        for key, value in GraphData.items():
            d = GraphData[key]
            if key == 'data_1': 
                ax.plot(d['Segments'], d['Values'], linestyle='-.', color='red')
                if g1lab:
                    ax.scatter(d['Segments'], d['Values'], marker='d', label='{}'.format(g1lab), color='red') 
                else:
                    ax.scatter(d['Segments'], d['Values'], marker='d', label='group 1', color='red')    
            else: 
                ax.plot(d['Segments'], d['Values'], color='blue')
                if g2lab:
                    ax.scatter(d['Segments'], d['Values'], s=75, label='{}'.format(g2lab), color='blue')  
                else:
                    ax.scatter(d['Segments'], d['Values'], s=75, label='group 2', color='blue')  

        # Horizontal line on X axis
        ax.axhline(y=0, color='black', linestyle='--')

        # Calculate the maximum absolute value of y for setting limits
        y_max = max(abs(min(GraphData['data_1']['Values'])), abs(min(GraphData['data_2']['Values'])), abs(max(GraphData['data_1']['Values'])), abs(max(GraphData['data_2']['Values'])))
        plt.ylim(-y_max, y_max)
        
        # Add regions labels if regions labels are specified
        if addRegion:
            for specRegion in addRegion:
                plt.axvline(x=specRegion[1], color='r', linestyle='--', label='{} at x={}'.format(specRegion[0], specRegion[1]))
                plt.text(specRegion[1], y_max + specRegion[3], '{} ({})'.format(specRegion[0], specRegion[1]), color='r', ha=specRegion[2])

        for dataN, dataV in GraphData.items():
            for index, p_value in enumerate(dataV['p_values']):
                region_y_psc = dataV['AVG_g1'][index]
                region_o_psc = dataV['AVG_g2'][index]
                region_diff_psc = dataV['Values'][index]
                if p_value < 0.001 and abs(region_y_psc - region_o_psc) > 1 and region_y_psc > 0 and region_o_psc > 0:
                    # print('sig ***')
                    annotation = '***'
                elif p_value < 0.01 and abs(region_y_psc - region_o_psc) > 1 and region_y_psc > 0 and region_o_psc > 0:
                    # print('sig **')
                    annotation = '**'
                elif p_value < 0.05 and abs(region_y_psc - region_o_psc) > 1 and region_y_psc > 0 and region_o_psc > 0:
                    # print('sig *')
                    annotation = '*'
                else: 
                    annotation = '' 
                if annotation:
                    # print('Index: {} | Data name: {} | Significance: {} | Y: {} | O: {} | Diff: {} | P_value: {}'.format(index, dataN, annotation, region_y_psc, region_o_psc, region_diff_psc, p_value))  
                    if dataN == 'data_1':
                        plt.annotate(annotation, 
                            (dataV['Segments'][index], dataV['Values'][index]),
                            textcoords="offset points", xytext=(0,10), ha='center', color='red')
                    else:
                        plt.annotate(annotation, 
                            (dataV['Segments'][index], dataV['Values'][index]),
                            textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

        plt.legend()
        plt.tight_layout()
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f'directory = {directory}, directory made')
            plt.savefig('{}\Diff_{}.svg'.format(directory, p), format='svg')
        plt.close()


# For visualising a single dataset at a time or visualising and comparing 2 datasets
# Takes a list of dataframes that have the same datastructure - column names etc. Will visualise every unique protein in the data
# If comparing 2 dataframes if the protein is present in both datasets then they will be compared

# requires: d = data (dataframe) for a particular protein or entire dataset of all proteins if protein specified,
#   diffColName = name of the column with average normalised difference in peptide spectral count (string),
#   pValueColName = name of the column containing p_value (string),
#   g1AverageColName = name of the group 1 average peptide spectral count column (string),
#   g2AverageColName = name of the group 2 average peptide spectral count column (string),
#   groupSampleNames = names of the columns containing the samples for comparison groups (list of lists containing strings), eg. groupSampleNames = [['FZ1_F (ms-status_RAT1)', 'FZ2_F (ms-status_RAT2)', 'FZ3_F (ms-status_RAT3)'], ['JA2_F (ms-status_RAT8)', 'JA5_F (ms-status_RAT11)', 'JA6_F (ms-status_RAT12)']], [['FZ4_M (ms-status_RAT4)', 'FZ5_M (ms-status_RAT5)', 'FZ6_M (ms-status_RAT6)'], ['JA1_M (ms-status_RAT7)', 'JA3_M (ms-status_RAT9)', 'JA4_M (ms-status_RAT10)']]

def visualiseAll(data, diffColName, pValueColName, g1AverageColName, g2AverageColName, groupSampleNames, *args, **kwargs):
    Thresh = kwargs.get('thresh', 0.05)

    Save = kwargs.get('save', None)   
    if Save:
        Save = True
    else:
        Save = False
    sigOnly = kwargs.get('sigOnly', None)
    directory = kwargs.get('directory', None)
    proteinSet = kwargs.get('sigonly', None)

    # Defines colour palette to be used for the heatmaps 
    colors = [(1, 1, 1), (1, 0, 0)]  # Transition from white (1, 1, 1) to red (1, 0, 0)
    n_bins = 100  # Number of discrete steps in the colormap
    cmap_name = 'white_to_red'  # Name of the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    if len(data) > 1:
        datasets = {}

        for i, d in enumerate(data):
            datasets[f'dataset_{i+1}'] = {
                'data':d,
                'samples':groupSampleNames[i]
        }
        
        all_prot_data_groups = []
        sig_prot_data_groups = []

        # add standard deviation
        for d in datasets:
            datasets[d]['data'] = addStandardDeviation(datasets[d]['data'], datasets[d]['samples'])
            pcol = next((s for s in datasets[d]['data'].columns if s.startswith('p:')), None)
            if pcol:
                datasets[d]['data'] = datasets[d]['data'].rename(columns={pcol: 'p:values'})
            # find significant proteins in both datasets
            datasets[d]['filtered_data'] = filterData(datasets[d]['data'], g1AverageColName, g2AverageColName, 'p:values', thresh=Thresh)

            all_prot_data_groups = all_prot_data_groups + list(datasets[d]['data']['Uniprot_ID'].unique())
            sig_prot_data_groups = sig_prot_data_groups + list(np.unique(datasets[d]['filtered_data']['prots']))

        # Define all datagroups, which proteins are unique to which datasets
        # empty list to store datasets while performing loop
        
        all_prot_data_groups = list(np.unique(np.array(all_prot_data_groups)))
        sig_prot_data_groups = list(np.unique(np.array(sig_prot_data_groups)))

        if sigOnly: 
            proteinSet = sig_prot_data_groups
        else:
            proteinSet = all_prot_data_groups

        for p in proteinSet:
            relevant_ds_names = []
            relevant_ds_df = []
            for index, ds in enumerate(datasets):
                if p in datasets[ds]['data']['Uniprot_ID'].unique():
                    relevant_ds_names.append(ds)
            if len(relevant_ds_names) > 1:
                for rds in relevant_ds_names:
                    data = datasets[rds]
                    data = data['data']
                    data = data[data["Uniprot_ID"] == p]
                    diff = next((s for s in data.columns if s.startswith('Diff')), None)
                    p_val = next((s for s in data.columns if s.startswith('p:')), None)
                    buildGraph(data, g1AverageColName, g1AverageColName, g2AverageColName, p_val, save=Save, directory=directory)
                    buildGraph(data, g2AverageColName, g1AverageColName, g2AverageColName, p_val, save=Save, directory=directory)
                for dn in relevant_ds_names:
                    tempdata = datasets[dn]['data']
                    relevant_ds_df.append(tempdata[tempdata['Uniprot_ID'] == p])
                visualiseDiffComparisons(relevant_ds_df, diff, p_val, g1AverageColName, g2AverageColName, save=Save, directory=directory, thresh=Thresh)
            else:
                data = datasets[relevant_ds_names[0]]
                data = data['data']
                data = data[data['Uniprot_ID'] == p]
                diff = next((s for s in data.columns if s.startswith('Diff')), None)
                p_val = next((s for s in data.columns if s.startswith('p:')), None)
                buildGraph(data, g1AverageColName, g1AverageColName, g2AverageColName, p_val, save=Save, directory=directory)
                buildGraph(data, g2AverageColName, g1AverageColName, g2AverageColName, p_val, save=Save, directory=directory)
                buildDiffGraph(data, diff, p_val, g1AverageColName, g2AverageColName, save=Save, directory=directory, thresh=Thresh)
    # If there is only 1 dataset run else        
    else:
        d = data[0]
        
        d = addStandardDeviation(d, groupSampleNames) # add standard deviation
        
        if sigOnly:
            significant_proteins = filterData(d, g1AverageColName, g2AverageColName, pValueColName) # filter for proteins with at least 1 significant region of PSM difference
            proteinSet = significant_proteins['prots']
        else:
            proteinSet = d['Uniprot_ID'].unique()

        for p in proteinSet:
            data = d[d['Uniprot_ID'] == p]
            for domain in data['Domain Type'].unique():
                data = data[data['Domain Type'] == domain]
                buildGraph(data, g1AverageColName, g1AverageColName, g2AverageColName, pValueColName, save=Save, directory=directory)
                buildGraph(data, g2AverageColName, g1AverageColName, g2AverageColName, pValueColName, save=Save, directory=directory)
                buildDiffGraph(data, diffColName, pValueColName, g1AverageColName, g2AverageColName, save=Save, directory=directory, thresh=Thresh)

    
    # data for difference comparison graphs
    # data = [data_f[data_f['Uniprot_ID'].isin(significant_proteins)], data_m[data_m['Uniprot_ID'].isin(significant_proteins)]]