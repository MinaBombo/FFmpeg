/*
 * Copyright (c) 2018 Mina Sami
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Color constancy filter
 *
 * @see http://colorconstancy.com/
 */

#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include <math.h>

#define TAG "\n[ColorConstancyFilter] "
typedef struct ColorConstancyContext {
    const AVClass *class;

    int difford;
    int minknorm;

    int nb_threads;
    int planeheight[4];
    int planewidth[4];

    double white[3];
} ColorConstancyContext;

#define OFFSET(x) offsetof(ColorConstancyContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption colorconstancy_options[] = {
    { "difford",  "set differentiation order", OFFSET(difford),  AV_OPT_TYPE_INT, {.dbl=1}, 0, 2, FLAGS },
    { "minknorm", "set Minkowski norm",        OFFSET(minknorm), AV_OPT_TYPE_INT, {.dbl=1}, 0, 9, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(colorconstancy);

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_GBRP,
        AV_PIX_FMT_NONE
    };

    return ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
}

static int config_props(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx = inlink->dst;
    ColorConstancyContext *s = ctx->priv;

    s->planewidth[1] = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0] = s->planewidth[3] = inlink->w;
    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;

    s->nb_threads = ff_filter_get_nb_threads(ctx);

    return 0;
}

typedef struct ThreadData {
    AVFrame *in;
    AVFrame *out;
    double *result;
    double minknorm;
} ThreadData;

static int diagonal_transformation(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    AVFrame *out = td->out;
    int plane;

    for (plane = 0; plane < 3; ++plane) {
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int64_t numpixels = width * (int64_t)height;
        const int slice_start = (numpixels * jobnr) / nb_jobs;
        const int slice_end = (numpixels * (jobnr+1)) / nb_jobs;
        const uint8_t *src = in->data[plane];
        uint8_t *dst = out->data[plane];
        double temp;
        unsigned i;

        for (i = slice_start; i < slice_end; ++i) {
            temp = (double)src[i] / s->white[plane];
            dst[i] = av_clip_uint8((int)(temp+0.5));
        }
    }
    return 0;
}

static void chromatic_adaptation(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData td;
    int nb_jobs = FFMIN3(s->planeheight[1], s->planewidth[1], s->nb_threads);

    td.in = in;
    td.out = out;
    ctx->internal->execute(ctx, diagonal_transformation, &td, NULL, nb_jobs);
}

static void normalize_light(AVFilterContext *ctx)
{
    ColorConstancyContext *s = ctx->priv;
    double *light = s->white;
    double abs_val = pow( pow(light[0], 2.0)+pow(light[1], 2.0)+pow(light[2], 2.0), 0.5);
    unsigned plane;

    if(!abs_val) {
        for (plane = 0; plane < 3; ++plane)
            light[plane] = 1.0;
    } else {
        for (plane = 0; plane < 3; ++plane) {
            light[plane] = (light[plane] / abs_val) * pow(3.0, 0.5);
            if (!light[plane])
                light[plane] = 1.0;
        }
    }
}

#define CLAMP(x, mx) ((x) < 0 ? 0 : ((x) >= (mx) ? (mx - 1) : (x)))
#define INDEX(i, dr, dc, w, h) CLAMP((i/w) + dr, h) * w + CLAMP(i%w + dc, w) 
#define DIFFX(s, i, w, h) s[INDEX(i, -1, -1, w, h)] *  1 + s[INDEX(i, 0, -1, w, h)] *  2 + s[INDEX(i, 1, -1, w, h)] *  1 + \
                          s[INDEX(i, -1,  1, w, h)] * -1 + s[INDEX(i, 0,  1, w, h)] * -2 + s[INDEX(i, 1,  1, w, h)] * -1 
#define DIFFY(s, i, w, h) s[INDEX(i, -1, -1, w, h)] *  1 + s[INDEX(i, -1, 0, w, h)] *  2 + s[INDEX(i, -1, 1, w, h)] *  1 + \
                          s[INDEX(i,  1, -1, w, h)] * -1 + s[INDEX(i,  1, 0, w, h)] * -2 + s[INDEX(i,  1, 1, w, h)] * -1 
#define DIFF(d, s, i, w, h) \
    int dx = DIFFX(s, i, w, h); \
    int dy = DIFFY(s, i, w, h); \
    d = av_clip_uint8(sqrt(dx*dx + dy*dy));
static int sobel(AVFilterContext* ctx, void* arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in  = td->in;
    AVFrame *out = td->out;
    int plane;

    for (plane = 0; plane < 3; ++plane) {
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int64_t numpixels = width * (int64_t)height;
        const int slice_start = (numpixels * jobnr) / nb_jobs;
        const int slice_end = (numpixels * (jobnr+1)) / nb_jobs;
        const uint8_t *src = in->data[plane];
        uint8_t *dst = out->data[plane];
        int i;

        for (i = slice_start; i < slice_end; ++i){
            DIFF(dst[i], src, i, width, height)
        }
    }
    return 0;
}

static int grey_constancy_preprocessing(AVFilterContext* ctx, void* arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    double *result = td->result;
    double minknorm = td->minknorm;
    int plane;

    for (plane = 0; plane < 3; ++plane) {
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int64_t numpixels = width * (int64_t)height;
        const int slice_start = (numpixels * jobnr) / nb_jobs;
        const int slice_end = (numpixels * (jobnr+1)) / nb_jobs;
        const uint8_t *data = in->data[plane];
        uint8_t d; 
        unsigned result_index = plane*nb_jobs + jobnr;
        int i;

        for (i = slice_start; i < slice_end; ++i){
             if(s->difford > 0) {
                    DIFF(d, data, i, width, height)
                } else 
                    d = data[i];

            if (s->minknorm > 0 )
                result[result_index] += (pow((double)d, minknorm));
            else 
                result[result_index] = FFMAX(result[result_index], d);
        }
    }
    return 0;
}

/* Operation:
    Max-RGB             : difford=0, minknorm=0, gblur=not-applied
    Grey-World          : difford=0, minknorm=1, gblur=not-applied
    Shades of Grey      : difford=0, minknorm=x, gblur=not-applied
    General Grey-World  : difford=0, minknorm=x, gblur=applied
    Max-Edge            : difford=1, minknorm=0, gblur=applied
    1st Order Grey-Edge : difford=1, minknorm=x, gblur=applied
    2nd Order Grey-Edge : difford=2, minknorm=x, gblur=applied
*/
static int filter_grey_constancy(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData td;
    unsigned nb_jobs = FFMIN3(s->planeheight[1], s->planewidth[1], s->nb_threads);
    unsigned plane, job;

    td.in = in;
    if (s->difford == 2) {
        td.out = out;
        ctx->internal->execute(ctx, sobel , &td, NULL, nb_jobs);
        td.in = out;
    }

    td.result = av_malloc_array(3, nb_jobs * sizeof(*td.result));
    if (!td.result) 
        return AVERROR(ENOMEM);
    for (plane=0; plane<3; ++plane) {
        s->white[plane] = 0.0;
        for (job=0; job<nb_jobs; ++job) {
            td.result[plane*nb_jobs + job] = 0;
        }
    }
    td.minknorm = s->minknorm;
    ctx->internal->execute(ctx, grey_constancy_preprocessing , &td, NULL, nb_jobs);

    if(s->minknorm > 0) {
        for (plane=0; plane<3; ++plane) {
            for (job=0; job<nb_jobs; ++job)
                s->white[plane] += td.result[plane*nb_jobs + job];
            s->white[plane] = pow(s->white[plane], 1.0/(double)s->minknorm);
        }
    } else {
         for (plane=0; plane<3; ++plane) 
            for (job=0; job<nb_jobs; ++job)
                s->white[plane] = FFMAX(s->white[plane] , td.result[plane*nb_jobs + job]);
    }

    av_freep(&td.result);
    return 0;
}

static int illumination_estimation(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    ColorConstancyContext *s = ctx->priv;
    int ret;
    ret = filter_grey_constancy(ctx, in, out);

    normalize_light(ctx);
    av_log(ctx, AV_LOG_INFO, "%sEstimated illumination= %f %f %f\n",
           TAG, s->white[0], s->white[1], s->white[2]);
    return ret;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;
    int ret;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out)
        return AVERROR(ENOMEM);
    av_frame_copy_props(out, in);
    av_frame_copy(out, in);

    if ((ret=illumination_estimation(ctx, in, out)))
        return ret;

    //chromatic_adaptation(ctx, in, out);

    return ff_filter_frame(outlink, out);
}

static const AVFilterPad colorconstancy_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad colorconstancy_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_colorconstancy = {
    .name          = "colorconstancy",
    .description   = NULL_IF_CONFIG_SMALL("Corrects image colors."),
    .priv_class    = &colorconstancy_class,
    .priv_size     = sizeof(ColorConstancyContext),
    .query_formats = query_formats,
    .inputs        = colorconstancy_inputs,
    .outputs       = colorconstancy_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};