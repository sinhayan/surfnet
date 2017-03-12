function gim = perform_sgim_sampling_fast(signal_original, pos_sphere, face, n, sampling_type)

% perform_sgim_sampling - generate a geometry image from a spherical parameterization
%
%   gim = perform_sgim_sampling(signal_original, pos_sphere, face, n);
%
%   'signal_original' are typically vertex location of the original mesh, but it can 
%   also be any information you want to re-sample on a regular grid (normal, etc).
%   'pos_sphere' are location on the sphere for these points.
%   'face' is the face data structure of the mesh.
%   'n' is width of the GIM.
%
%   Uses the spherical geometry image datastructure introduced in
%       Spherical parametrization and remeshing.
%       E. Praun, H. Hoppe.
%       SIGGRAPH 2003, 340-349.
%
%   Copyright (c) 2004 Gabriel PeyrÈ

if nargin<4
	n = 64;
end

if nargin<5
    sampling_type = 'area';
end

if size(signal_original,2)~=size(pos_sphere,2)
    error('Original and spherical meshes must be of same size.');
end

% compute sampling location on the image
% disp('Computing planar sampling locations.');
posw = perform_spherial_planar_sampling(pos_sphere, sampling_type);

% perform 4-fold symmetry 
sym = { [0,1], [0,-1], [1, 0], [-1, 0] };
for i=1:length(sym)
    c = sym{i};
    if c(1)==0
        I = find(posw(2,:)*sign(c(2))>=0);
    else
        I = find(posw(1,:)*sign(c(1))>=0);
    end
    posi = posw(:,I);
    posi = perform_symmetry(posi,c);
    posw = [posw, posi];
    signal_original = [signal_original, signal_original(:,I)];
end

% crop a bit to speed up ..
m = 1.5;
I = find( posw(1,:)<m & posw(1,:)>-m & posw(2,:)<m & posw(2,:)>-m);
posw = posw(:,I);
signal_original = signal_original(:,I);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERPOLATE THE CENTER (orignal triangulation)
% remove the faces that cross the boundary
% edges = [face(1:2,:), face(2:3,:), face([1 3],:)];
% pos = pos_sphere;
% J = find( pos(3,edges(1,:))<=0 & pos(3,edges(2,:))<=0 &     ...
%           ( pos(1,edges(1,:)).*pos(1,edges(2,:))<=0   |     ...
%             pos(2,edges(1,:)).*pos(2,edges(2,:))<=0 )  );
% % find the corresponding faces
% J = unique( mod(J-1,size(face,2))+1 );
% % find the complementary
% x = zeros(size(face,2),1); x(J) = 1; 
% I = x==0;
% % retrive face number
% face1 = face; 
% face1 = face1(:,I);
% Index_face=I;
% compute the interpolation on this sub-set using original triangulation
% posn = (n-1)*(posw+1)/2+1;      % sampling location in [1,n]≤
% gim1 = zeros( n, n, size(signal_original,1) );
% fprintf('Griding using original triangulation ');
% for i=1:size(signal_original, 1)
% %     fprintf('.');
%     gim1(:,:,i) = griddata_arbitrary( face1, posn, signal_original(i,:), n);
% end


% remove doublons
if 1
[tmp,I] = unique(posw(1,:)+pi*posw(2,:));
posw = posw(:,I);
signal_original = signal_original(:,I);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERPOLATE THE BOUNDARY (delaunay triangulation)
% sampling location in [-1,1]
x = -1:2/(n-1):1;
[Y,X] = meshgrid(x,x);
% interpolate location
gim2 = zeros( n, n, size(signal_original,1) );
% fprintf('\nGriding using Delaunay triangulation ');

% DT = delaunayTriangulation([posw(1,:)', posw(2,:)']);
% DT.ConnectivityList=face1';
% DT.Points=[posw(1,:)', posw(2,:)'];

for i=1:size(signal_original, 1)
%     fprintf('.');
%     gim2(:,:,i) = griddata( posw(1,:), posw(2,:), signal_original(i,:), X, Y );
%     Fscatter = TriScatteredInterp(posw(1,:)', posw(2,:)',signal_original(i,:)');
      Fscatter = scatteredInterpolant(posw(1,:)', posw(2,:)',signal_original(i,:)','natural');
%     F = TriScatteredInterp(DT, signal_original(i,:)');
    gim2(:,:,i)=Fscatter(X,Y);
end
% fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % MIX THE TWO
% gim = gim1;
% I = find( isnan(gim) );
% gim(I) = gim2(I);
% 
% I =  isnan(gim) ;
% gim(I) = 0;

% to keep delaunay uncomment this
   gim = gim2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% translate/scale to fit in a box [0,1]^3
% for i=1:size(signal_original, 1)
%     x = gim(:,:,i); x = x(:);
%     m = min( x );
%     gim(:,:,i) = gim(:,:,i) - m;
% end
%gim = rescale(gim);

function y = perform_symmetry(x,c)
% y = 2*c - x
y(1,:) = 2*c(1) - x(1,:);
y(2,:) = 2*c(2) - x(2,:);

function posw = perform_spherial_planar_sampling(pos_sphere, sampling_type)

% perform_spherial_planar_sampling - project sampling location from sphere to a square
%
% posw = perform_spherial_planar_sampling(pos_sphere, type)
%
%   'type' can be 'area' or 'gnomonic'.
%
%   This is used to produced spherical geometry images.
%   The sampling is first projected onto an octahedron and then unfolded on
%   a square.
%
%   Copyright (c) 2004 Gabriel Peyre

if nargin<2
    sampling_type = 'area';
end

% all 3-tuple of {-1,1}
a = [-1 1];
[X,Y,Z] = meshgrid(a,a,a);

anchor_2d = {};
anchor_3d = {};
for i=1:8
   	x = X(i); y = Y(i); z = Z(i);
   	a2d = []; a3d = [];
	if z>0
		a2d = [a2d, [0;0]];
    else
		a2d = [a2d, [x;y]];
    end
    a2d = [a2d, [x;0]];
    a2d = [a2d, [0;y]];
    anchor_2d{i} = a2d;    
    a3d = [a3d, [0;0;z]];
    a3d = [a3d, [x;0;0]];
    a3d = [a3d, [0;y;0]];
    anchor_3d{i} = a3d;
end

pos = pos_sphere;
n = size(pos, 2);
posw = zeros(2,n);
for s = 1:8
    x = X(s); y = Y(s); z = Z(s);
    anc2d = anchor_2d{s};
    anc3d = anchor_3d{s};
    I = find( signe(pos(1,:))==x & signe(pos(2,:))==y & signe(pos(3,:))==z );
    posI = pos(:,I);
    nI = length(I);
    if strcmp(sampling_type, 'area')
        % find the area of the 3 small triangles
        p1 = repmat(anc3d(:,1), 1, nI);
        p2 = repmat(anc3d(:,2), 1, nI);
        p3 = repmat(anc3d(:,3), 1, nI);
        a1 = compute_spherical_area( posI, p2, p3 );
        a2 = compute_spherical_area( posI, p1, p3 );
        a3 = compute_spherical_area( posI, p1, p2 );
        % barycentric coordinates
        a = a1+a2+a3;
        % aa = compute_spherical_area( p1, p2, p3 );
        a1 = a1./a; a2 = a2./a; a3 = a3./a;
    elseif strcmp(sampling_type, 'gnomonic')
        % we are searching for a point y=b*x (projection on the triangle)
        % such that :   a1*anc3d(:,1)+a2*anc3d(:,2)+a3*anc3d(:,3)-b*x=0
        %               a1+a2+a3=1
        a1 = zeros(1,nI); a2 = a1; a3 = a1;
        for i=1:nI
            x = posI(:,i);
            M = [1 1 1 0; anc3d(:,1), anc3d(:,2), anc3d(:,3), x];
            a = M\[1;0;0;0];
            a1(i) = a(1);
            a2(i) = a(2);
            a3(i) = a(3);
        end
    else
        error('Unknown projection method.');
    end
    posw(:,I) = anc2d(:,1)*a1 + anc2d(:,2)*a2 + anc2d(:,3)*a3;
end


function y = signe(x)

y = double(x>=0)*2-1;

function A = compute_spherical_area( p1, p2, p3 )

% length of the sides of the triangles :
% cos(a)=p1*p2
a  = acos( dotp(p2,p3) );
b  = acos( dotp(p1,p3) );
c  = acos( dotp(p1,p2) );
s  = (a+b+c)/2;
% use L'Huilier's Theorem
% tand(E/4)^2 = tan(s/2).*tan( (s-a)/2 ).*tan( (s-b)/2 ).*tan( (s-c)/2 )
E = tan(s/2).*tan( (s-a)/2 ).*tan( (s-b)/2 ).*tan( (s-c)/2 );
A = 4*atan( sqrt( E ) );
A = real(A);

function d = dotp(x,y)
d = x(1,:).*y(1,:) + x(2,:).*y(2,:) + x(3,:).*y(3,:);

function M = griddata_arbitrary(face,vertex,v,n, options)

% griddata_arbitrary - perform interpolation of a triangulation on a regular grid
%
%   M = griddata_arbitrary(face,vertex,v,n);
%
% 'n' is the size of the image
% 'vertex' are assumed to lie in [1,n]^2
%
%   Copyright (c) 2004 Gabriel Peyré

options.null = 0;
% verb = getoptions(options, 'verb', 1);
verb = 0;

if nargin<4
    error('4 arguments required.');
end
if size(face,2)==3 && size(face,1)~=3
    face = face';
end
if size(face,1)~=3
    error('Works only with triangulations.');
end
if size(vertex,2)==2 && size(vertex,1)~=2
    vertex = vertex';
end
if size(vertex,1)~=2
    error('Works only with 2D triangulations.');
end

if min(vertex(:))>=0 && max(vertex(:))<=1
    % date in [0,1], makes conversion
    vertex = vertex*(n-1)+1;
end

nface = size(face,2);
face_sampling = 0;
if length(v)==nface
    face_sampling = 1;
end

M = zeros(n);
Mnb = zeros(n);
for i=1:nface
    if verb
%         progressbar(i,nface);
    end
    T = face(:,i);          % current triangles
    P = vertex(:,T);        % current points
    V = v(T);               % current values
    % range
    selx = min(floor(P(1,:)))-1:max(ceil(P(1,:)))+1;
    sely = min(floor(P(2,:)))-1:max(ceil(P(2,:)))+1;
    % grid locations
    [Y,X] = meshgrid(sely,selx);
    pos = [X(:)'; Y(:)'];
    p = size(pos,2);    % number of poitns
    % solve barycentric coords
    c = [1 1 1; P]\[ones(1,p);pos];
    % find points inside the triangle
    I = find( c(1,:)>=-10*eps & c(2,:)>=-10*eps & c(3,:)>=-10*eps );
    pos = pos(:,I);
    c = c(:,I);
    % restrict to point inside the image
    I = find( pos(1,:)<=n & pos(1,:)>0 & pos(2,:)<=n & pos(2,:)>0 );
    pos = pos(:,I);
    c = c(:,I);
    % convert sampling location to index in the array
    J = sub2ind(size(M), pos(1,:),pos(2,:) );
    % assign values
    if ~isempty(J)
        if face_sampling
            M(J) = v(i);
        else
            M(J) = M(J) + V(1)*c(1,:) + V(2)*c(2,:) + V(3)*c(3,:);
        end
        Mnb(J) = Mnb(J)+1;
    end
end
I = find(Mnb>0);
M(I) = M(I)./Mnb(I);
I = find(Mnb==0);
M(I) = NaN;

function progressbar(n,N,w)

% progressbar - display a progress bar
%
%    progressbar(n,N,w);
%
% displays the progress of n out of N.
% n should start at 1.
% w is the width of the bar (default w=20).
%
%   Copyright (c) Gabriel Peyré 2006

if nargin<3
    w = 20;
end

% progress char
cprog = '.';
cprog1 = '*';
% begining char
cbeg = '[';
% ending char
cend = ']';

p = min( floor(n/N*(w+1)), w);

global pprev;
if isempty(pprev)
    pprev = -1;
end

if not(p==pprev)
    ps = repmat(cprog, [1 w]);
    ps(1:p) = cprog1;
    ps = [cbeg ps cend];
    if n>1
        % clear previous string
        fprintf( repmat('\b', [1 length(ps)]) );
    end
    fprintf(ps);
end
pprev = p;
if n==N
    fprintf('\n');
end

function v = getoptions(options, name, v, mendatory)

% getoptions - retrieve options parameter
%
%   v = getoptions(options, 'entry', v0);
% is equivalent to the code:
%   if isfield(options, 'entry')
%       v = options.entry;
%   else
%       v = v0;
%   end
%
%   Copyright (c) 2007 Gabriel Peyre

if nargin<4
    mendatory = 0;
end

if isfield(options, name)
    v = eval(['options.' name ';']);
elseif mendatory
    error(['You have to provide options.' name '.']);
end 



