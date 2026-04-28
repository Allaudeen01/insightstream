import React from 'react';

interface InsightCardProps {
  title: string;
  impact: 'High' | 'Medium' | 'Low';
  description: string;
  qualified_segments?: string[];
  recommendation?: string;
  rule_type?: string;
}

const InsightCard = ({ 
  title, 
  impact, 
  description, 
  qualified_segments,
  recommendation,
  rule_type
}: InsightCardProps) => {
  const impactColors = {
    High:   'bg-red-500/20 text-red-300 border-red-500/30',
    Medium: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    Low:    'bg-blue-500/20 text-blue-300 border-blue-500/30',
  };

  return (
    <div className="
      rounded-xl border border-white/10
      bg-white/5 backdrop-blur-sm p-6
      hover:border-purple-500/30
      hover:shadow-lg hover:shadow-purple-500/5
      transition-all duration-200
    ">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <h3 className="font-semibold text-lg text-foreground leading-tight pr-4">
          {title}
        </h3>
        <span className={`
          text-xs font-medium px-2.5 py-1 
          rounded-full border shrink-0
          ${impactColors[impact]}
        `}>
          {impact} Impact
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        {description}
      </p>

      {/* Qualified segments */}
      {qualified_segments && qualified_segments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {qualified_segments.map(seg => (
            <span key={seg} className="
              text-xs px-3 py-1 rounded-full
              bg-purple-500/20 text-purple-300
              border border-purple-500/30
              font-medium
            ">
              {seg}
            </span>
          ))}
        </div>
      )}

      {/* Recommendation */}
      {recommendation && (
        <div className="
          mt-4 pt-4 border-t border-white/10
          flex gap-3 items-start
        ">
          <span className="text-purple-400 mt-0.5 shrink-0 font-bold">
            →
          </span>
          <p className="text-sm text-purple-300 leading-relaxed">
            {recommendation}
          </p>
        </div>
      )}
    </div>
  );
};

export default InsightCard;
