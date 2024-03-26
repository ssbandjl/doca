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
#pragma once

#include "fabric.h"
#include "reserved_string.h"
#include "sub_tree_info.h"
#include "sub_tree_score.h"

#define TREE_PREFIX         "tree "
#define TREE_HASH_PREFIX    "Tree hash "
#define NODE_PREFIX         "node "
#define SUB_NODE_PREFIX     "subNode "
#define COMPUTE_PORT_PREFIX "computePort "

using MapPortKeyToTreeNodes = std::multimap<uint64_t, class TreeNode*>;
using TreeNodesVec = std::vector<SetTreeNodePtr>;

class AggTree
{
    sharp_trees_t m_tree_id_;
    uint8_t m_max_rank_;
    bool m_is_valid_;
    bool m_is_configured_;
    SetTreeNodePtrSortedByANGuid m_tree_nodes_;
    TreeNode* m_root_tree_node_;
    MapPortKeyToTreeNodes m_compute_to_tree_nodes_;
    SetPortKey m_disconnected_compute_ports_;
    SetAnFabricInfoPtr m_span_leafs;   // leaf span by this tree
    AggTree* m_sat;                    // SAT tree duplicated from this tree;
    AggTree* m_llt;                    // LLT tree duplication source of this tree;
    sharp_resource_priority m_priority_;
    uint64_t m_tree_hash_;

   public:
    AggTree(sharp_trees_t tree_id);
    ~AggTree();

    inline const SetTreeNodePtrSortedByANGuid& GetTreeNodes() const { return m_tree_nodes_; }
    inline sharp_trees_t GetId() const { return m_tree_id_; }
    inline sharp_resource_priority GetPriority() const { return m_priority_; }
    inline void SetPriority(sharp_resource_priority priority) { m_priority_ = priority; }
    inline TreeNode* GetRootTreeNode() { return m_root_tree_node_; };
    inline void SetRootTreeNode(TreeNode* tree_node) { m_root_tree_node_ = tree_node; }
    inline AggTree* GetSat() { return m_sat; }
    inline void SetSat(AggTree* p_sat) { m_sat = p_sat; }
    inline AggTree* GetLlt() { return m_llt; }
    inline void SetLlt(AggTree* p_llt) { m_llt = p_llt; }
    inline bool IsSat() const { return m_llt != NULL; }
    inline uint64_t GetHash() const { return m_tree_hash_; };
    inline void SetHash(const uint64_t new_tree_hash) { m_tree_hash_ = new_tree_hash; };
    inline bool IsValid() const { return m_is_valid_; }
    inline bool IsConfigured() const { return m_is_configured_; }

    // Create TreeNode objects for specified AggNodes and set parent-child
    // relation between them.
    //
    // switch_roots enables connecting two sub trees even if p_sub_agg_node is
    // not the root of the second sub tree.
    // When switch_roots is enabled, the second sub tree is rotated to make
    // p_sub_agg_node its root, and then connects it as a child of p_agg_node.
    //
    // This method modifies AggNode objects.
    int CreateTreeConnection(AggNodeFabricInfo* p_agg_node, AggNodeFabricInfo* p_sub_agg_node, bool switch_roots = false);

    // create tree node with no Descendant
    int CreateTreeNode(AggNodeFabricInfo* p_agg_node);

    int ConfigureOnFabric();
    int RemoveConfigurationFabric();

    int SetRoot();
    int SetRoot(AggNodeFabricInfo* p_agg_node);
    int SetTreeNodesRank();

    int DuplicateTree(const AggTree* p_src_tree);
    TreeNode* DuplicateSubTree(TreeNode* p_src_node);

    bool IsNodesSupportsSat();

    int UpdateComputeToTreeNode(MapPortKeyToAggNodeFabrics& compute_por_to_agg_nodes, bool skip_if_exist);

    int CanSubTreeMatchJobRequest(const SetPortDataConstPtr& compute_ports,
                                  const JobResource& job_resource,
                                  JobSubTreeInfo& job_sub_tree_info,
                                  const JobSubTreeScore& default_tree_score,
                                  JobSubTreeScore& sub_tree_result,
                                  const sharp_job_id_t sharp_job_id);

    int PrepareSubTreeForJob(JobSubTreeInfo& job_sub_tree_info);

    int UpdateTreeTree();
    int UpdateDfpTree(const SetAnFabricInfoPtr& span_leafs_diff);
    int BuildTree(AggNodeFabricInfo* p_tree_root, const SetAnFabricInfoPtr& descendant_leafs, const AggNodeFabricInfo* parent_agg_node);

    void DumpTree(std::string& dump_message) const;
    void DumpState(FILE* f) const;

    bool IsComputePortMappedtToTreeNode(const uint64_t port_key) const;

    void ClearComputeToTreeNode(port_key_t compute_port_key);

    int SelectMostFreeAggPaths();
    void CalculateHash();

   private:
    int UpdateTreeNodesVec(TreeNodesVec& tree_nodes_vec,
                           uint8_t& min_tree_rank,
                           uint8_t& max_tree_rank,
                           TreeNode* p_tree_node,
                           const PortData* p_compute_port,
                           uint8_t child_index,
                           JobSubTreeScore& sub_tree_result,
                           JobSubTreeInfo& sub_tree_info);

    // Rotate subtree until specified TreeNode object becomes root
    int RotateSubTree(TreeNode* p_tree_node);

    // During BuildTree span all non leaf ANs if required
    int SpanUnconnectedDescendants(AggNodeFabricInfo* p_sub_root_fabric_info);
};
