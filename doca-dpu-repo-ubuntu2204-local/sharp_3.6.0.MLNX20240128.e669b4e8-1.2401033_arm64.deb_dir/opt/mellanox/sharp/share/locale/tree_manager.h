/*
 * Copyright (c) 2012-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef TREE_MANAGER_H_
#define TREE_MANAGER_H_

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "agg_types.h"
#include "option_manager.h"
#include "port_data.h"
#include "sub_tree_info.h"
#include "sub_tree_score.h"

class Fabric;
class AggNode;
class AggNodeFabricInfo;
class AggTree;
struct PathScore;
struct AnScore;
struct Permutation;
struct DescendantGroup;
struct QuasiFatTreeAnInfo;
struct BestRouteInfo;

typedef std::vector<AnScore> VecAnScore;
typedef std::vector<PathScore> VecPathScore;
typedef std::unordered_set<AggNodeFabricInfo*> SetANInfo;
typedef std::set<TreeNode*> SetTreeNodePtr;
typedef std::vector<std::unique_ptr<TreeNode>> TreeNodesVecUniquePtr;
typedef std::queue<AggNodeFabricInfo*> AggNodeFabricInfoQ;
typedef std::vector<QuasiFatTreeAnInfo*> QuasiFatTreeAnInfoVec;
typedef std::vector<DescendantGroup*> VecPtrDescendantGroup;
typedef std::list<DescendantGroup> ListDescendantGroup;
typedef std::unordered_map<AggNodeFabricInfo*, QuasiFatTreeAnInfo> MapANToQuasiFatTreeInfo;
typedef std::unordered_map<AggNodeFabricInfo*, BestRouteInfo> MapANToBestRouteInfo;
typedef std::unordered_map<AggNodeFabricInfo*, int> MapANInfoToCount;
typedef std::unordered_map<uint64_t, DescendantGroup*> MapHashToDescendantGroup;

struct DescendantGroup
{
    JobSubTreeScore score;
    SetANInfo leaf_nodes;
    AggNodeFabricInfo* child_node;
    VecPtrDescendantGroup combined_by_groups;
    uint64_t group_hash;

    DescendantGroup(AggNodeFabricInfo* child_an) : child_node(child_an), group_hash(0) {}
};

struct QuasiFatTreeAnInfo
{
    AggNodeFabricInfo* agg_node;
    MapHashToDescendantGroup raw_desc_groups;
    DescendantGroup* best_group;
    MapANInfoToCount map_valid_path_count_to_child;

    QuasiFatTreeAnInfo(AggNodeFabricInfo* an) : agg_node(an) { best_group = nullptr; };
};

struct BestRouteInfo
{
    AggNodeFabricInfo* parent_node;
    AggNodeFabricInfo* child_node;
    JobSubTreeScore* score;

    BestRouteInfo(AggNodeFabricInfo* parent, AggNodeFabricInfo* child, JobSubTreeScore* job_sub_tree_score)
        : parent_node(parent), child_node(child), score(job_sub_tree_score)
    {}
};

class TreeManager
{
    SetAnFabricInfoPtr m_dfp_leafs_;   // Used only for DFP topology

   public:
    TreeManager() {}

    int BuildTrees(bool seamless_restart);
    int UpdateTrees();

    int BuildSatTrees();
    int BuildDynamicQuasiFatTree(const SetAnFabricInfoPtr& leaf_nodes_set,
                                 JobSubTreeInfo& job_sub_tree_info,
                                 const JobResource& job_resource,
                                 JobSubTreeScore& sub_tree_score,
                                 TreeNodesVecUniquePtr& tmp_tree_nodes_vec);

    int DumpTrees() const;
    int DumpTreesState() const;

   private:
    int ParseFabricTreesFile(FILE* f);

    // parse the node desc section of trees file
    AggNodeFabricInfo* ParseAggNodeInfo(const char* line, int line_num);

    // set Tree root: the tree node with no parent.
    // validate tree structure: one and only one root
    int SetTreesRoot();
    int SetTreesNodesRank();

    int ParseFabricTreesFile(bool is_seamless_restart);
    int CalculateTrees();

    int CalculateTreeTrees();
    int UpdateTreeTrees();

    // BFS Trees calculation functions
    int CreateBfsTree(AggTree* p_agg_tree, AggNodeFabricInfo* p_agg_node);
    int CalculateBfsTrees();

    // Hyper-Cube trees calculation functions
    int CalculateHyperCubeTrees();
    static int CreateHyperCubeKruskalTree(AggTree* p_agg_tree,
                                          AggNodeFabricInfo* p_root_agg_node,
                                          VecPathScore& path_score_vector,
                                          Permutation perm);
    static void KruskalTreeUnifySets(AggNodeFabricInfo* p_agg_node1,
                                     AggNodeFabricInfo* p_agg_node2,
                                     std::vector<int>& an_set_vector,
                                     std::vector<std::set<int>>& set_id_set_vector);
    static bool KruskalTreeCheckSets(AggNodeFabricInfo* p_agg_node1,
                                     AggNodeFabricInfo* p_agg_node2,
                                     std::vector<int>& an_set_vector,
                                     std::vector<std::set<int>>& set_id_set_vector);

    // Dragonfly Plus Trees calculation functions
    int CalculateDfpTrees();
    int UpdateDfpTrees();
    void SetDfpTreeHeightAndLeafsNumber();
    void SortDfpRoots(vector<uint8_t>& groups_root_load);

    void ClearDescendantLeafs();
    void BuildDescendantLeafs();
    uint16_t GetMaxDescendants(bool& is_partial_spanning);

    static int ParseNodeInfo(const char* line, int line_num, std::string& node_desc, uint64_t& port_guid);

    void SortTreeRootsByGroup();
    void UpdateDfpLeafs();
    const SetAnFabricInfoPtr& GetDfpLeafs() { return m_dfp_leafs_; }

    int QuasiFatTreeBFSBottomUp(AggNodeFabricInfoQ& agg_node_info_q,
                                MapANToQuasiFatTreeInfo& agg_node_to_quasi_fat_tree_info,
                                QuasiFatTreeAnInfoVec& quasi_fat_tree_an_info_of_roots,
                                JobSubTreeInfo& job_sub_tree_info,
                                const JobResource& job_resource,
                                ListDescendantGroup& all_descendant_groups,
                                TreeNodesVecUniquePtr& tmp_tree_nodes_vec,
                                const SetAnFabricInfoPtr& leaf_nodes_set);

    int QuasiFatTreeBFSUpBottom(QuasiFatTreeAnInfo* root_quasi_an_info,
                                JobSubTreeInfo& job_sub_tree_info,
                                MapANToQuasiFatTreeInfo& agg_node_to_quasi_fat_tree_info,
                                const JobResource& job_resource,
                                JobSubTreeScore& sub_tree_score);

    int QuasiFatTreeSelectBestRoute(MapANToBestRouteInfo& leaves_route,
                                    JobSubTreeInfo& job_sub_tree_info,
                                    MapANToQuasiFatTreeInfo& agg_node_to_quasi_fat_tree_info,
                                    QuasiFatTreeAnInfoVec& next_rank_quasi_fat_tree_an_info,
                                    SetANInfo& used_agg_nodes);

    bool CheckIfAggNodeSatisfyJobRequest(const AggNodeFabricInfo& agg_node, const JobResource& job_resource);

    void CombineGroups(QuasiFatTreeAnInfo& quasi_fat_tree_info,
                       VecPtrDescendantGroup& combined_groups,
                       JobSubTreeInfo& job_sub_tree_info,
                       ListDescendantGroup& all_descendant_groups);

    void CombineGroupsWithRawGroup(VecPtrDescendantGroup& combined_groups,
                                   DescendantGroup& raw_group,
                                   JobSubTreeInfo& job_sub_tree_info,
                                   QuasiFatTreeAnInfo& quasi_fat_tree_info,
                                   ListDescendantGroup& all_descendant_groups);

    int UpdateQuasiFatTreeJobSubTreeInfo(JobSubTreeInfo& job_sub_tree_info, JobSubTreeScore& sub_tree_score);
};

#endif   // TREE_MANAGER_H_
