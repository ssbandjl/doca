/*
 * Copyright (c) 2004-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under the terms of the
 * OpenIB.org BSD license included below:
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

#ifndef AGG_REG_EXP_H
#define AGG_REG_EXP_H

#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>
#include <string>

using namespace std;

/*
 * This file holds simplified object oriented interface for regular expressions
 */

// TODO update and use ibis reg_exp
namespace am_tools
{

/////////////////////////////////////////////////////////////////////////////
class RexMatch
{
   private:
    const char* str;
    size_t nMatches;
    regmatch_t* matches;

    RexMatch(const RexMatch& other);
    const RexMatch& operator=(const RexMatch& rhs);

    // no default constructor
    RexMatch();

   public:
    // construct:
    RexMatch(const char* s, size_t numMatches)
    {
        str = s;
        nMatches = numMatches;
        matches = new regmatch_t[nMatches + 1];
    };

    // destrutor:
    ~RexMatch() { delete[] matches; };

    // useful:
    int field(size_t num, char* buf)
    {
        // check for non match:
        if (num > nMatches || matches[num].rm_so < 0) {
            buf[0] = '\0';
            return 0;
        }
        strncpy(buf, str + matches[num].rm_so, matches[num].rm_eo - matches[num].rm_so + 1);
        buf[matches[num].rm_eo - matches[num].rm_so] = '\0';
        return 0;
    };

    // string returning the field
    int field(size_t num, string& res)
    {
        string tmp = string(str);
        // check for non match:
        if (num > nMatches || matches[num].rm_so < 0) {
            res = "";
            return 0;
        }
        res = tmp.substr(matches[num].rm_so, matches[num].rm_eo - matches[num].rm_so);
        return 0;
    };

    string field(size_t num)
    {
        string tmp = string(str);
        // check for non match:
        if (num > nMatches || matches[num].rm_so < 0) {
            return string("");
        }
        return tmp.substr(matches[num].rm_so, matches[num].rm_eo - matches[num].rm_so);
    };

    size_t numFields() { return nMatches; }

    friend class RegExp;
};

/////////////////////////////////////////////////////////////////////////////
// This will basically hold the compiled regular expression
class RegExp
{
   private:
    regex_t re;
    char* expr;
    int status;

    RegExp(const RegExp& other);
    const RegExp& operator=(const RegExp& rhs);
    // no default constructor
    RegExp();

   public:
    // construct
    RegExp(const char* pattern, int flags = REG_EXTENDED) : expr(NULL), status(1)
    {
        status = 1;
        expr = new char[strlen(pattern) + 1];
        strcpy(expr, pattern);
        status = regcomp(&re, expr, flags);
    }

    // destructor
    ~RegExp()
    {
        regfree(&re);
        delete[] expr;
    }

    // access:
    const char* getExpr() const { return expr; }
    int valid() const { return (status == 0); }

    // usefullnss:
    class RexMatch* apply(const char* str, int flags = 0) const
    {
        if (status)
            return NULL;
        class RexMatch* res = new RexMatch(str, (int)re.re_nsub);
        if (regexec(&re, str, re.re_nsub + 1, res->matches, flags)) {
            delete res;
            return 0;
        }
        return res;
    }

    friend class RexMatch;
};

}   // namespace am_tools

#endif /* AGG_REG_EXP_H */
