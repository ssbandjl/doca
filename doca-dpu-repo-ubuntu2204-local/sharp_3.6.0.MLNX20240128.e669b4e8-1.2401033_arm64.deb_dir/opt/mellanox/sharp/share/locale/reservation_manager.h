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

#ifndef RESERVATION_MANAGER_H_
#define RESERVATION_MANAGER_H_

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "agg_ib_types.h"
#include "option_manager.h"
#include "smx/smx_api.h"

class CommandManager;
class SharpJob;
class ReservationJobInfo;
class AggTree;

// Max number of jobs we can expect at most extreme case is as number of nodes divide by 2
#define MAX_RESERVATION_LIMIT (FABRIC_MAX_VALID_LID / 2)

typedef std::map<uint64_t, ReservationJobInfo> JobInfos;   // Client JobID to ReservationJobInfo
typedef std::unordered_set<port_key_t> HashPortKeys;

typedef std::vector<unsigned int> ResourceLimitValueByRank;   // The index in the vector is the nodes rank (0 = root, max = leaf),
                                                              // the value is the limit that a reservation can use

class ReservationJobInfo
{
    uint64_t m_client_job_id_;
    HashPortKeys m_guids;

   public:
    ReservationJobInfo() : m_client_job_id_(0){};
    ~ReservationJobInfo(){};

    void SetClientJobId(uint64_t client_job_id) { m_client_job_id_ = client_job_id; }

    uint64_t GetClientJobId() const { return m_client_job_id_; }

    bool IsGuidsEmpty() const { return m_guids.empty(); }

    void InsertGuid(port_key_t port_key) { m_guids.insert(port_key); }

    bool CheckAtLeastOneGuidInJob(std::vector<port_key_t>& check_guids);
};

class ReservationInfo
{
    struct sharp_reservation_info m_info_;
    JobInfos m_job_infos_;
    uint64_t m_reservation_id_;   // temporary member, should be removed when reservation_id removed

   public:
    ResourceLimitValueByRank m_resource_limit_remaining_by_rank;

    ReservationInfo();

    ReservationInfo(const sharp_reservation_info* ri);

    ~ReservationInfo();

    void SetReservationInfo(const sharp_create_reservation& cr);

    void UpdateReservationInfo(const sharp_create_reservation& cr);

    void SetResourceLimitation(const ResourceLimitValueByRank& base_value_by_rank);

    void SetState(enum sharp_reservation_state state) { m_info_.state = state; }

    const char* GetKey() const { return m_info_.reservation_key; }

    uint16_t GetPkey() const
    {
        return m_info_.pkey;   // return only PKEY, without membership bit
    }

    uint16_t GetFullPkey() const
    {
        return m_info_.pkey | 0x8000;   // Return PKEY with full membership. Force using full PKEY for the reservation
    }

    enum sharp_reservation_state GetState() const { return m_info_.state; }

    port_key_t GetGuid(uint64_t index) const { return m_info_.port_guids[index]; }

    uint32_t GetNumGuids() const { return m_info_.num_guids; }

    const struct sharp_reservation_resources& GetInfoResources() const { return m_info_.resource_limitations; }

    const struct sharp_reservation_info& GetInfo() const { return m_info_; }

    bool IsJobEmpty() const { return m_job_infos_.empty(); }

    void RemoveJob(JobInfos::iterator& it);

    void RemoveJob(const SharpJob* job);

    void AddJob(SharpJob* job);

    void ApplyJobReservationResourcesChange(const SharpJob* p_job, bool is_added_job);

    bool IsJobIdsEnd(JobInfos::iterator& it) const { return (it == m_job_infos_.end()); }

    JobInfos::iterator GetJobIdsIter() { return m_job_infos_.begin(); }

    uint64_t GetReservationId() { return m_reservation_id_; }

    void SetReservationId(uint64_t reservation_id) { m_reservation_id_ = reservation_id; }

    enum sharp_reservation_status CheckModifiedResources(const sharp_create_reservation& create_reservation);
};

class ReservationManager
{
    typedef std::map<string, ReservationInfo*> MapKeyToReservationInfo;
    typedef std::map<uint64_t, ReservationInfo*> MapGuidToReservationInfo;
    typedef std::unordered_map<uint64_t, string> MapIdToReservationKey;
    typedef std::pair<port_key_t, ReservationInfo*> PairGuidToReservationInfo;
    typedef std::pair<string, ReservationInfo*> PairKeyToReservationInfo;
    typedef std::pair<uint64_t, string> PairIdToReservationKey;

    CommandManager* const m_command_manager_;

    MapKeyToReservationInfo m_reservations_;
    MapGuidToReservationInfo m_guid_to_reservations_;
    MapIdToReservationKey m_id_to_reservation_key_;

   public:
    ResourceLimitValueByRank m_resource_limit_base_value_by_rank;

    ReservationManager(CommandManager* const p_command_manager) : m_command_manager_(p_command_manager) {}

    ~ReservationManager();

    void CreateReservation(const sharp_create_reservation& create_reservation, const smx_ep* ep, uint64_t tid);

    void DeleteReservation(const sharp_delete_reservation& delete_reservation, const smx_ep* ep, uint64_t tid);

    void ReservationInfoRequest(const sharp_reservation_info_request& request, const smx_ep* ep, uint64_t tid);

    ReservationInfo* GetReservationByJob(const SharpJob* p_job);

    ReservationInfo* GetReservationByGuids(uint64_t* guids, uint32_t num_guids);

    void PrepareReservationResourceLimit(const AggTree* agg_tree);

    void PrepareReservationResourceWithoutLimit(uint8_t max_sw_rank);

    ReservationInfo* GetReservationInfoByKey(const std::string& reservation_key);

    void ReadPersistentReservationInfoFiles();

   private:
    enum sharp_reservation_status InternalUpdateReservation(ReservationInfo* ri, const sharp_create_reservation& create_reservation);
    void InternalCreateNewReservation(ReservationInfo* ri);
    void InternalDeleteReservation(ReservationInfo* ri);
    void RemoveReservationJobs(ReservationInfo* ri);
    void CreateReservationId(ReservationInfo& ri);
    void SetReservationId(ReservationInfo& ri, uint64_t ri_id);

    void AddReservationToDB(ReservationInfo* reservation);

    void CreateReservationInfoFile(const sharp_reservation_info& reservation_info, const string& persistent_path);

    void DeleteReservationInfoFile(const sharp_reservation_info& reservation_info, const string& persistent_path);

    void DeleteFile(const string& file_path);

    void UpdateReservationInfoFile(const sharp_reservation_info& reservation_info, const string& persistent_path);

    int ChechIfDirExists(const string& dir_name) const;

    int BuildDataFromReservationFile(const char* job_file, sharp_smx_msg*& smx_msg) const;

    void LogResourceLimitBaseValues();
};
#endif   // RESERVATION_MANAGER_H_
