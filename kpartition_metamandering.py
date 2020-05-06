# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:15:58 2020

@author: lnajt
"""


#use gerrychain.tree.recursive_tree_part(graph, parts, pop_target, pop_col, epsilon, node_repeats=1, method=<function bipartition_tree>

#gerrychain.tree.bipartition_tree(graph, pop_col, pop_target, epsilon, node_repeats=1, spanning_tree=None, choice=<bound method Random.choice of <random.Random object>>)


def find_gerrymander(graph , num_parts):
    
    



steps = 200
ns = 1
m = 10

pop1 = .01
#widths = [0,.5,1,1.5,2,2.5,3]
#widths = [1,2,3]
#widths = [1.5]
widths = [0]
chaintype = "uniform_tree"
#chaintype = "tree"
p = .6
proportion = p*6
#####
#widths = [0,.5]
#widths = [1,1.5]
#widths = [3]
print("proportion:", proportion)
#for p in [.6,.55,.65,.7, .75]:
diagonal_bias = "debiased"

diagonal_bias = "four_squares"
tree_types = ["uniform_tree", "tree"]
diagonal_bias = "anti_four_squares"
diagonal_bias = "center_square"
diagonal_bias = "one_line"
diagonal_bias = "random_mander"
p_1 = .7
p_2 = .4

noise = False



for trial in range(10):
    print("trial:", trial)
    for p_2 in [.3]:
        for p_diff in [.25]:
            p_1 = min(1, p_2 + p_diff)
    
            
            for chaintype in ["uniform_tree", "tree"]:
                print(chaintype)
                widths = [2.5]
                for p in [.6]:
                    proportion = p * 6
                    for width in widths:
            

                        if noise == True:
                            graph = random_mandering(m, biased_parition, p_1, p_2)
                        if noise == False:
                            graph = metamander(biased_partition)
                            
                            
                        plt.figure()
                        nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
                        plt.savefig("./plots/Attractor/" + "_trial_" + str(trial)  +  "_p_1" + str(p_1) +  "_p_2" + str(p_2) +  "Size" + str(m) + "WIDTH" + str(width) + "Bias" + str(diagonal_bias) +  "UnderlyingGraph.png" )
                        plt.close()
            
                        horizontal = []
                        for x in graph.nodes():
                            if x[1] < 6 * m / 2:
                                horizontal.append(x)
                        vertical = []
                        for x in graph.nodes():
                            if x[1] < 3 * m:
                                vertical.append(x)
            
            
                        cddict = {}  # {x: 1-2*int(x[0]/gn)  for x in graph.nodes()}
            
                        start_plans = [horizontal]
                        alignment = 0
                        for n in graph.nodes():
                            if n in start_plans[alignment]:
                                cddict[n] = 1
                            else:
                                cddict[n] = -1
            
                        for edge in graph.edges():
                            graph[edge[0]][edge[1]]['cut_times'] = 0
            
                        for n in graph.nodes():
                            graph.nodes[n]["population"] = 1
                            graph.nodes[n]["part_sum"] = cddict[n]
                            graph.nodes[n]["last_flipped"] = 0
                            graph.nodes[n]["num_flips"] = 0
            
                            if n[0] == 0 or n[0] == m - 1 or n[1] == m or n[1] == -m + 1:
                                graph.nodes[n]["boundary_node"] = True
                                graph.nodes[n]["boundary_perim"] = 1
            
                            else:
                                graph.nodes[n]["boundary_node"] = False
            
            
                        ####CONFIGURE UPDATERS
            
                        def new_base(partition):
                            return base
            
            
                        def step_num(partition):
                            parent = partition.parent
            
                            if not parent:
                                return 0
            
                            return parent["step_num"] + 1
            
            
                        bnodes = [x for x in graph.nodes() if graph.nodes[x]["boundary_node"] == 1]
            
            
                        def bnodes_p(partition):
                            return [x for x in graph.nodes() if graph.nodes[x]["boundary_node"] == 1]
            
            
                        updaters = {'population': Tally('population'),
                                    "boundary": bnodes_p,
                                    'cut_edges': cut_edges,
                                    'step_num': step_num,
                                    'b_nodes': b_nodes_bi,
                                    'base': new_base,
                                    'geom': geom_wait,
                                    # "Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                                    }
            
                        #########BUILD PARTITION
            
                        grid_partition = Partition(graph, assignment=cddict, updaters=updaters)
            
                        base = 1
                        # ADD CONSTRAINTS
                        popbound = within_percent_of_ideal_population(grid_partition, pop1)
                        '''
                        plt.figure()
                        nx.draw(graph, pos={x: x for x in graph.nodes()}, node_size=ns,
                                node_shape='s', cmap='tab20')
                        plt.savefig("./plots/Attractor/" + "_trial_" + str(trial)  + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) +    "B" + str(int(100 * base)) + "P" + str(
                            int(100 * pop1)) + "start.png" )
                        plt.close()'''
            
                        #########Setup Proposal
                        ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)
            
                        tree_proposal = partial(recom,
                                                pop_col="population",
                                                pop_target=ideal_population,
                                                epsilon=pop1,
                                                node_repeats=1
                                                )
            
                        #######BUILD MARKOV CHAINS
                        if chaintype == "flip":
                            exp_chain = MarkovChain(slow_reversible_propose_bi,
                                                    Validator([single_flip_contiguous, popbound  # ,boundary_condition
                                                               ]), accept=cut_accept, initial_state=grid_partition,
                                                    total_steps=steps)
            
            
                        if chaintype == "tree":
                            tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                                    node_repeats=1, method=my_mst_bipartition_tree_random)
            
                            exp_chain = MarkovChain(tree_proposal,
                                                    Validator([popbound  # ,boundary_condition
                                                               ]), accept=cut_accept, initial_state=grid_partition,
                                                    total_steps=steps)
            
                        if chaintype == "uniform_tree":
                            #tree_proposal = partial(uniform_tree_propose)
                            tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                                    node_repeats=1, method=my_uu_bipartition_tree_random)
            
                            exp_chain = MarkovChain(tree_proposal,
                                                    Validator([popbound  # ,boundary_condition
                                                               ]), accept=cut_accept, initial_state=grid_partition,
                                                    total_steps=steps)
            
                        #########Run MARKOV CHAINS
            
                        rsw = []
                        rmm = []
                        reg = []
                        rce = []
                        rbn = []
                        waits = []
            
                        import time
            
                        st = time.time()
            
                        t = 0
                        seats = [[],[]]
                        vote_counts = [[],[]]
                        old = 0
                        #skip = next(exp_chain)
                        #skip the first partition
                        k = 0
                        num_cuts_list = []
                        for part in exp_chain:
                            if k > 0:
                                #if part.assignment == old:
                                #    print("didn't change")
                                rce.append(len(part["cut_edges"]))
                                waits.append(part["geom"])
                                rbn.append(len(list(part["b_nodes"])))
                                num_cuts = len(part["cut_edges"])
                                num_cuts_list.append(num_cuts)
                                for edge in part["cut_edges"]:
                                    graph[edge[0]][edge[1]]["cut_times"] += 1
                                    # print(graph[edge[0]][edge[1]]["cut_times"])
            
                                if part.flips is not None:
                                    f = list(part.flips.keys())[0]
                                    graph.nodes[f]["part_sum"] = graph.nodes[f]["part_sum"] - dict(part.assignment)[f] * (
                                        abs(t - graph.nodes[f]["last_flipped"]))
                                    graph.nodes[f]["last_flipped"] = t
                                    graph.nodes[f]["num_flips"] = graph.nodes[f]["num_flips"] + 1
                                for i in [0, 1]:
                                    top = []
                                    bottom = []
                                    for n in graph.nodes():
                                        if part.assignment[n] == 1:
                                            top.append( int(n[i] < proportion*m))
                                        if part.assignment[n] == -1:
                                            bottom.append( int( n[i] < proportion*m))
            
                                    top_seat = int(np.mean(top) > .5)
                                    bottom_seat = int(np.mean(bottom) > .5)
                                    total_seats = top_seat + bottom_seat
                                    seats[i].append(total_seats)
                                #old = part.assignment
                            t += 1
                            k += 1
                        print("average cut size", np.mean(num_cuts_list))
                        f = open("./plots/Attractor/" + "_trial_" + str(trial)  +  "_p_1" + str(p_1) +  "_p_2" + str(p_2) + str(alignment) + "SAMPLES" + str(steps) + "Size" + str(m) + "chaintype" + str(chaintype) + "Bias" + str(diagonal_bias) + "P" + str(
                            int(100 * pop1)) + "proportion" + str(p) + "edges.txt", 'a')
            
                        means = np.mean(seats,1)
                        stds = np.std(seats,1)
            
                        f.write( "_p_1: " + str(p_1) +  " p_2: " + str(p_2) + "  " + str( means[0] ) + "(" + str(stds[0]) + ")," + str( means[1] ) + "(" + str(stds[1]) + ")" + "at width:" + str(width) + '\n')
            
                        #f.write("mean:" +  str(np.mean(seats,1)) + "var:" + str(np.var(seats,1)) + "stdev:" + str(np.std(seats,1)) +  "at width:" + str(width) + '\n' )
            
                        f.close()
                        print("_p_1: " + str(p_1) +  " p_2: " + str(p_2) + "  " + str( means[0] ) + "(" + str(stds[0]) + ")," + str( means[1] ) + "(" + str(stds[1]) + ")" )
                        #print(seats)
            
                        plt.figure()
                        nx.draw(graph, pos={x: x for x in graph.nodes()}, node_color=[0 for x in graph.nodes()], node_size=1,
                                edge_color=[graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape='s',
                                cmap='magma', width=3)
                        plt.savefig("./plots/Attractor/" + "_trial_" + str(trial)  +  "_p_1" + str(p_1) +  "_p_2" + str(p_2) + str(alignment) + "SAMPLES" + str(steps) + "Size" + str(m) + "WIDTH" + str(width) + "chaintype" +str(chaintype) +  "Bias" + str(diagonal_bias) +  "P" + str(
                            int(100 * pop1)) + "edges.png" )
                        plt.close()
            
                        A2 = np.zeros([6 * m, 6 * m])
                        for n in graph.nodes():
                            #print(n[0], n[1] - 1, dict(part.assignment)[n])
                            A2[n[0], n[1]] = dict(part.assignment)[n]
            
                        plt.figure()
                        plt.imshow(A2, cmap = 'jet')
                        plt.axis('off')
                        plt.savefig("./plots/Attractor/" + "_trial_" + str(trial)  +  "_p_1" + str(p_1) +  "_p_2" + str(p_2) + "Size" + str(m) + "WIDTH" + str(width) + "chaintype" +str(chaintype) + "Bias" + str(diagonal_bias) + "P" + str(
                            int(100 * pop1)) + "sample_partition.png" )
                        plt.close()
            
                        #plt.figure()
                        #plt.hist(seats)