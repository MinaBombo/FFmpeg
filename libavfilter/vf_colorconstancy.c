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
 * 
 * @algorithm: grey_edge 
 * Based on "Edge-Based Color Constancy"
 * by J. van de Weijer, T. Gevers, A. Gijsenij
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

#define NUM_PLANES 3
#define MAX_DIFF_ORD 2
typedef struct ColorConstancyContext {
    const AVClass *class;

    int difford;
    int minknorm;
    float sigma;

    int filtersize;
    double *gauss[MAX_DIFF_ORD+1];

    int nb_threads;
    int planeheight[4];
    int planewidth[4];

    double white[3];
} ColorConstancyContext;

#define OFFSET(x) offsetof(ColorConstancyContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption colorconstancy_options[] = {
    { "difford",  "set differentiation order", OFFSET(difford),  AV_OPT_TYPE_INT,   {.dbl=1},   0,   2,      FLAGS },
    { "minknorm", "set Minkowski norm",        OFFSET(minknorm), AV_OPT_TYPE_INT,   {.dbl=1},   0,   65535,  FLAGS },
    { "sigma",    "set sigma",                 OFFSET(sigma),    AV_OPT_TYPE_FLOAT, {.dbl=0.5}, 0.0, 1024.0, FLAGS },
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

#define FINDX(s, i) ((s) + (i))
static int set_gauss(AVFilterContext *ctx)
{
    ColorConstancyContext *s = ctx->priv;
    int filtersize = s->filtersize;
    int sigma = s->sigma;
    double sum1, sum2;
    int i;

    for (i = 0; i <= MAX_DIFF_ORD; ++i)  
        s->gauss[i] = NULL;

    for (i = 0; i <= MAX_DIFF_ORD; ++i) {
        s->gauss[i] = av_malloc_array(2*filtersize+1, sizeof(*s->gauss[i]));
        if(!s->gauss[i]) {
            for (i = 0; i <= MAX_DIFF_ORD; ++i) 
                av_freep(&s->gauss[i]);
            return AVERROR(ENOMEM);
        }
    }

    for (i = -filtersize; i <= filtersize; ++i)
        s->gauss[0][FINDX(filtersize, i)] = 1/(sqrt(2 * M_PI) * sigma)* exp(pow(i, 2.)/(-2 * sigma * sigma) );

    sum1 = 0.0;
    for (i = -filtersize; i <= filtersize; ++i) {
        s->gauss[1][FINDX(filtersize, i)] = -(i/pow(sigma, 2))*s->gauss[0][FINDX(filtersize, i)];
        sum1 += s->gauss[1][FINDX(filtersize, i)] * i;
    }

    for (i = 0; i <= 2*filtersize ; ++i) 
        s->gauss[1][i] /= sum1;

    sum1 = 0.0;
    for (i = -filtersize; i <= filtersize; ++i) {
        s->gauss[2][FINDX(filtersize, i)] = (pow(i, 2)/pow(sigma, 4) - 1/pow(sigma, 2)) * s->gauss[0][FINDX(filtersize, i)];
        sum1 += s->gauss[2][FINDX(filtersize, i)];
    }

    sum2 = 0.0;
    for (i = -filtersize; i <= filtersize; ++i) {
        s->gauss[2][FINDX(filtersize, i)] -= sum1/(2*filtersize+1);
        sum2 += (0.5*i*i*s->gauss[2][FINDX(filtersize, i)]);
    }
    for (i = 0; i <= 2*filtersize ; ++i) 
        s->gauss[2][i] /= sum2;

    return 0;
}

#define BREAK_OFF_SIGMA  3.0
static int config_props(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    ColorConstancyContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    int ret;
    
    s->filtersize = floor(BREAK_OFF_SIGMA*s->sigma+0.5);
    if(ret=set_gauss(ctx))
        return ret;

    s->nb_threads = ff_filter_get_nb_threads(ctx);
    s->planewidth[1] = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0] = s->planewidth[3] = inlink->w;
    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;

    return 0;
}

#define MAX_META_DATA 4
#define MAX_DATA 4
typedef struct ThreadData {
    AVFrame *in, *out;
    int meta_data[MAX_META_DATA];
    double  *data[MAX_DATA][NUM_PLANES];
} ThreadData;

#define INDEX_TEMP 0
#define INDEX_DX   1
#define INDEX_DY   2
#define INDEX_DXY  3
#define INDEX_NORM INDEX_DX
#define INDEX_SRC 0
#define INDEX_DST 1
#define INDEX_ORD 2
#define INDEX_DIR 3

static int filter_slice_grey_edge(AVFilterContext* ctx, void* arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    double minknorm = s->minknorm;
    int plane;

    for (plane = 0; plane < NUM_PLANES; ++plane) {
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int64_t numpixels = width * (int64_t)height;
        const int slice_start = (numpixels * jobnr) / nb_jobs;
        const int slice_end = (numpixels * (jobnr+1)) / nb_jobs;
        const double *src = td->data[INDEX_NORM][plane];
        double *dst = td->data[INDEX_DST][plane];
        int i;

        if (s->minknorm > 0)
            for (i = slice_start; i < slice_end; ++i)
                dst[jobnr] += (pow(src[i]/255., minknorm));
        else
            for (i = slice_start; i < slice_end; ++i)
                dst[jobnr] = FFMAX(dst[i], src[i]);
    }
    return 0;
}

static void cleanup_derivative_buffers(AVFilterContext* ctx, ThreadData *td)
{
    int i, j;

    for (i = 0; i < MAX_DATA; ++i)
        for (j = 0; j < NUM_PLANES; ++j)
            av_freep(&td->data[i][j]);
}

static int setup_derivative_buffers(AVFilterContext* ctx, ThreadData *td)
{
    ColorConstancyContext *s = ctx->priv;
    int nb_buff = s->difford + 1;
    int i, j;

    for (i = 0; i < MAX_DATA; ++i)
        for (j = 0; j < NUM_PLANES; ++j)
            td->data[i][j] = NULL;

    for (i = 0; i <= nb_buff; ++i) {
        for (j = 0; j < NUM_PLANES; ++j) {
            td->data[i][j] = av_malloc_array(s->planeheight[1] * s->planewidth[1], sizeof(*td->data[i][j]));
            if (!td->data[i][j]){
                cleanup_derivative_buffers(ctx, td);
                return AVERROR(ENOMEM);
            }
        }
    }

    return 0;
}

#define DIR_X 0
#define DIR_Y 1
#define CLAMP(x, mx) ( (x) < 0 ? 0 : ((x)>=(mx)) ? (mx-1) : x ) 
static int slice_get_derivative(AVFilterContext* ctx, void* arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    const int ord = td->meta_data[INDEX_ORD];
    const int dir = td->meta_data[INDEX_DIR];
    const int src_index = td->meta_data[INDEX_SRC];
    const int dst_index = td->meta_data[INDEX_DST];
    const int filtersize = s->filtersize;
    const double *gauss = s->gauss[ord];
    int plane;

    for(plane = 0; plane < NUM_PLANES; ++plane){
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int ly = (dir == DIR_X ? height : width); 
        const int lx = (dir == DIR_X ? width  : height); 
        const int slice_start = (ly *  jobnr   ) / nb_jobs;
        const int slice_end   = (ly * (jobnr+1)) / nb_jobs;
        double *dst = td->data[dst_index][plane];
        const void *src;
        int y, x, z, linesize;
        int *r = (dir == DIR_X ? &y : &x), *c = (dir == DIR_X ? &x : &y);

        if (dir == DIR_X){
            src = in->data[plane]; 
            linesize = in->linesize[plane];
            for (y = slice_start; y < slice_end; ++y) 
                for (x = 0; x < lx; ++x) 
                    for (z = -filtersize; z <= filtersize; ++z)
                        dst[(*r)*width+(*c)] += (((uint8_t*)src)[ CLAMP((*r)+z*dir, height) * linesize + CLAMP((*c)+z*(1-dir), width)] * 
                                                gauss[FINDX(filtersize, z)]);
        } else {
            src = td->data[src_index][plane]; 
            linesize = width;
            for (y = slice_start; y < slice_end; ++y)
                for (x = 0; x < lx; ++x)
                    for (z = -filtersize; z <= filtersize; ++z)
                        dst[(*r)*width+(*c)] += (((double*)src)[ CLAMP((*r)+z*dir, height) * linesize + CLAMP((*c)+z*(1-dir), width)] *
                                                gauss[FINDX(filtersize, z)]);
        }
        
    }
    return 0;
}

static int slice_normalize(AVFilterContext* ctx, void* arg, int jobnr, int nb_jobs)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData *td = arg;
    const int difford = s->difford;
    int plane;

    for (plane = 0; plane < NUM_PLANES; ++plane) {
        const int height = s->planeheight[plane];
        const int width  = s->planewidth[plane];
        const int64_t numpixels = width * (int64_t)height;
        const int slice_start = (numpixels * jobnr) / nb_jobs;
        const int slice_end = (numpixels * (jobnr+1)) / nb_jobs;
        const double *x = td->data[INDEX_DX][plane];
        const double *y = td->data[INDEX_DY][plane];
        double *norm = td->data[INDEX_NORM][plane];
        int i;

        if(difford == 1)
            for (i = slice_start; i < slice_end; ++i)
                norm[i] = sqrt( pow(x[i], 2) + pow(y[i], 2));
        else {
            const double *z = td->data[INDEX_DXY][plane];
            for (i = slice_start; i < slice_end; ++i)
                norm[i] = sqrt( pow(x[i], 2) + 4 * pow(z[i], 2) + pow(y[i], 2));
        }

    }
    return 0;
}

static int get_normalized_derivative(AVFilterContext *ctx, AVFrame *in, ThreadData *td)
{
    ColorConstancyContext *s = ctx->priv;

    switch(s->difford){
        case 0:
            td->meta_data[INDEX_ORD] = 0; 

            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = 0;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = 0;
            td->meta_data[INDEX_DST] = 1;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));
            return 0;

        case 1:
            td->meta_data[INDEX_ORD] = 1; 
            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = INDEX_TEMP;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 0; 
            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = INDEX_TEMP;
            td->meta_data[INDEX_DST] = INDEX_DX;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 0; 
            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = INDEX_TEMP;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 1; 
            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = INDEX_TEMP;
            td->meta_data[INDEX_DST] = INDEX_DY;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));
            return 0;

        case 2:
            td->meta_data[INDEX_ORD] = 2; 
            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = INDEX_TEMP;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 0; 
            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = INDEX_TEMP;
            td->meta_data[INDEX_DST] = INDEX_DX;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 0; 
            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = INDEX_TEMP;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 2; 
            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = INDEX_TEMP;
            td->meta_data[INDEX_DST] = INDEX_DY;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));

            td->meta_data[INDEX_ORD] = 1; 
            td->meta_data[INDEX_DIR] = DIR_X;
            td->meta_data[INDEX_DST] = INDEX_TEMP;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planeheight[1], s->nb_threads));

            td->meta_data[INDEX_DIR] = DIR_Y;
            td->meta_data[INDEX_SRC] = INDEX_TEMP;
            td->meta_data[INDEX_DST] = INDEX_DXY;
            ctx->internal->execute(ctx, slice_get_derivative, td, NULL, FFMIN(s->planewidth[1], s->nb_threads));
            return 0;

        default:
            return AVERROR(EINVAL);
    }

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
static int filter_grey_edge(AVFilterContext *ctx, AVFrame *in)
{
    ColorConstancyContext *s = ctx->priv;
    ThreadData td;
    int nb_jobs = FFMIN3(s->planeheight[1], s->planewidth[1], s->nb_threads);
    int plane, job, ret;

    td.in = in;
    if (ret=setup_derivative_buffers(ctx, &td))
        return ret;
    get_normalized_derivative(ctx, in, &td);
    if (s->difford > 0)
        ctx->internal->execute(ctx, slice_normalize, &td, NULL, nb_jobs);


    ctx->internal->execute(ctx, filter_slice_grey_edge, &td, NULL, nb_jobs);
    if(s->minknorm > 0) {
        for (plane=0; plane<3; ++plane) {
            for (job=0; job<nb_jobs; ++job)
                s->white[plane] += td.data[INDEX_DST][plane][job];
            s->white[plane] = pow(s->white[plane], 1.0/(double)s->minknorm);
        }
    } else {
         for (plane=0; plane<3; ++plane) 
            for (job=0; job<nb_jobs; ++job)
                s->white[plane] = FFMAX(s->white[plane] , td.data[INDEX_DST][plane][job]);
    }
    
    cleanup_derivative_buffers(ctx, &td);
    return 0;
}

static void normalize_light(double *light)
{
    double abs_val = pow( pow(light[0], 2.0)+pow(light[1], 2.0)+pow(light[2], 2.0), 0.5);
    unsigned plane;

    if(!abs_val) {
        for (plane = 0; plane < 3; ++plane)
            light[plane] = 1.0;
    } else {
        for (plane = 0; plane < 3; ++plane) {
            light[plane] = (light[plane] / abs_val);
            if (!light[plane])
                light[plane] = 1.0;
        }
    }
}

static int illumination_estimation(AVFilterContext *ctx, AVFrame *in)
{
    ColorConstancyContext *s = ctx->priv;
    int ret;
    ret = filter_grey_edge(ctx, in);

    normalize_light(s->white);
    av_log(ctx, AV_LOG_INFO, "%sEstimated illumination= %f %f %f\n",
           TAG, s->white[0], s->white[1], s->white[2]);
    return ret;
}

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
            temp = (double)src[i] / (s->white[plane] * pow(3.0, 0.5));
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

    if ((ret=illumination_estimation(ctx, in))) {
        av_freep(&out);
        return ret;
    }
 
    chromatic_adaptation(ctx, in, out);

    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ColorConstancyContext *s = ctx->priv;
    int i;

    for (i = 0; i <= MAX_DIFF_ORD; ++i) 
        av_freep(&s->gauss[i]);
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
    .uninit        = uninit,
    .priv_class    = &colorconstancy_class,
    .priv_size     = sizeof(ColorConstancyContext),
    .query_formats = query_formats,
    .inputs        = colorconstancy_inputs,
    .outputs       = colorconstancy_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};